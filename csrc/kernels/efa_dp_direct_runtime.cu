/**
 * @file efa_dp_direct_runtime.cu
 * @brief Host-side runtime implementation for efa-dp-direct + DeepEP
 *
 * This file implements the EFA device initialization, QP management,
 * memory registration, and GPU-side state setup for GPU-direct RDMA
 * via efa-dp-direct on AWS EFA.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "efa_cuda_dp.h"
#include "efa_dp_direct_device.cuh"
#include "efa_dp_direct_runtime.cuh"
#include "exception.cuh"

// GPU-side global device state
__device__ deep_ep::efa_dp_device_state efa_dp_state_d;

namespace deep_ep {
namespace efa_dp_runtime {

// ============================================================================
// Module-level state
// ============================================================================

static struct {
    bool initialized = false;
    int rank = -1;
    int num_ranks = 0;
    int num_qps_per_peer = 0;

    // EFA resources
    struct ibv_context *ctx = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_cq *cq = nullptr;
    union ibv_gid gid;

    // QPs: [peer_idx * num_qps_per_peer + qp_id]
    std::vector<struct ibv_qp*> ibv_qps;
    std::vector<efa_cuda_qp*> gpu_qps;
    std::vector<struct efadv_wq_attr> sq_attrs;
    std::vector<struct efadv_wq_attr> rq_attrs;

    // AHs: one per peer
    std::vector<struct ibv_ah*> ahs;
    std::vector<uint16_t> ah_nums;

    // Memory registrations
    struct ibv_mr *main_mr = nullptr;
    void *main_buf = nullptr;
    size_t main_buf_size = 0;

    // Host-side copy of device state (to be memcpy'd to GPU)
    efa_dp_device_state host_state;

} g_state;

// ============================================================================
// Internal helpers
// ============================================================================

static struct ibv_device* find_efa_device() {
    int num_devices;
    struct ibv_device **devices = ibv_get_device_list(&num_devices);
    if (!devices) return nullptr;

    struct ibv_device *efa = nullptr;
    for (int i = 0; i < num_devices; i++) {
        // EFA devices have node_type == 7 (IBV_NODE_UNSPECIFIED)
        if (devices[i]->node_type == 7) {
            efa = devices[i];
            break;
        }
    }
    // Note: caller should not free device list until done
    return efa;
}

static int create_srd_qp(struct ibv_qp **out_qp, uint32_t qkey) {
    struct ibv_qp_init_attr_ex attr_ex = {};
    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = g_state.cq;
    attr_ex.recv_cq = g_state.cq;
    attr_ex.cap.max_send_wr = 256;
    attr_ex.cap.max_recv_wr = 256;
    attr_ex.cap.max_send_sge = 2;
    attr_ex.cap.max_recv_sge = 2;
    attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr_ex.pd = g_state.pd;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;

    struct efadv_qp_init_attr efa_attr = {};
    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;

    *out_qp = efadv_create_qp_ex(g_state.ctx, &attr_ex, &efa_attr, sizeof(efa_attr));
    if (!*out_qp) {
        fprintf(stderr, "efa_dp_runtime: efadv_create_qp_ex failed: %s\n", strerror(errno));
        return -1;
    }

    // Transition to RTS: INIT → RTR → RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = 1;
        a.qkey = qkey;
        if (ibv_modify_qp(*out_qp, &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
            fprintf(stderr, "efa_dp_runtime: QP INIT failed\n");
            return -1;
        }
    }
    {
        struct ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTR;
        if (ibv_modify_qp(*out_qp, &a, IBV_QP_STATE)) {
            fprintf(stderr, "efa_dp_runtime: QP RTR failed\n");
            return -1;
        }
    }
    {
        struct ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn = 0;
        if (ibv_modify_qp(*out_qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
            fprintf(stderr, "efa_dp_runtime: QP RTS failed\n");
            return -1;
        }
    }

    return 0;
}

static int register_qp_for_gpu(struct ibv_qp *qp, efa_cuda_qp **out_gpu_qp,
                                struct efadv_wq_attr *out_sq, struct efadv_wq_attr *out_rq) {
    // Query hardware pointers
    struct efadv_wq_attr sq = {}, rq = {};
    if (efadv_query_qp_wqs(qp, &sq, &rq, sizeof(sq))) {
        fprintf(stderr, "efa_dp_runtime: efadv_query_qp_wqs failed\n");
        return -1;
    }

    // Register SQ buffer and doorbell with GPU (requires root for IoMemory)
    cudaError_t err;

    err = cudaHostRegister(sq.buffer, (size_t)sq.entry_size * sq.num_entries,
                          cudaHostRegisterIoMemory);
    if (err != cudaSuccess) {
        fprintf(stderr, "efa_dp_runtime: cudaHostRegister SQ buffer failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Register SQ doorbell (page-aligned)
    void *sq_db_page = (void*)((uintptr_t)sq.doorbell & ~0xFFF);
    err = cudaHostRegister(sq_db_page, 4096, cudaHostRegisterIoMemory);
    if (err != cudaSuccess) {
        fprintf(stderr, "efa_dp_runtime: cudaHostRegister SQ doorbell failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Register RQ buffer (DMA memory, not IO)
    err = cudaHostRegister(rq.buffer, (size_t)rq.entry_size * rq.num_entries,
                          cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "efa_dp_runtime: cudaHostRegister RQ buffer failed: %s\n",
                cudaGetErrorString(err));
        // Non-fatal for send-only QPs
    }

    // Register RQ doorbell
    void *rq_db_page = (void*)((uintptr_t)rq.doorbell & ~0xFFF);
    err = cudaHostRegister(rq_db_page, 4096, cudaHostRegisterIoMemory);
    if (err != cudaSuccess) {
        // Non-fatal for send-only QPs
    }

    // Create efa-dp-direct GPU QP
    struct efa_cuda_qp_attrs qa = {};
    qa.sq_buffer = (uint8_t*)sq.buffer;
    qa.rq_buffer = (uint8_t*)rq.buffer;
    qa.sq_doorbell = sq.doorbell;
    qa.rq_doorbell = rq.doorbell;
    qa.sq_num_entries = sq.num_entries;
    qa.sq_entry_size = sq.entry_size;
    qa.sq_max_batch = sq.max_batch;
    qa.rq_num_entries = rq.num_entries;
    qa.rq_entry_size = rq.entry_size;
    qa.reserved = 0;

    *out_gpu_qp = efa_cuda_create_qp(&qa, sizeof(qa));
    if (!*out_gpu_qp) {
        fprintf(stderr, "efa_dp_runtime: efa_cuda_create_qp failed\n");
        return -1;
    }

    *out_sq = sq;
    *out_rq = rq;
    return 0;
}

// ============================================================================
// Public API
// ============================================================================

std::vector<uint8_t> get_unique_id() {
    // For OOB exchange, return our GID as unique ID
    // In practice, use torch.distributed for bootstrap
    std::vector<uint8_t> id(sizeof(union ibv_gid));
    if (g_state.ctx) {
        ibv_query_gid(g_state.ctx, 1, 0, &g_state.gid);
        std::memcpy(id.data(), &g_state.gid, sizeof(union ibv_gid));
    }
    return id;
}

int init(int rank, int num_ranks, int num_qps_per_peer, bool low_latency_mode) {
    if (g_state.initialized) return 0;

    g_state.rank = rank;
    g_state.num_ranks = num_ranks;
    g_state.num_qps_per_peer = num_qps_per_peer;

    fprintf(stderr, "[efa_dp_runtime] Initializing: rank=%d/%d, qps_per_peer=%d\n",
            rank, num_ranks, num_qps_per_peer);

    // 1. Open EFA device
    struct ibv_device *efa_dev = find_efa_device();
    if (!efa_dev) {
        fprintf(stderr, "efa_dp_runtime: No EFA device found\n");
        return -1;
    }

    g_state.ctx = ibv_open_device(efa_dev);
    if (!g_state.ctx) {
        fprintf(stderr, "efa_dp_runtime: ibv_open_device failed\n");
        return -1;
    }
    fprintf(stderr, "[efa_dp_runtime] EFA device: %s\n", ibv_get_device_name(efa_dev));

    // 2. Create PD and CQ
    g_state.pd = ibv_alloc_pd(g_state.ctx);
    g_state.cq = ibv_create_cq(g_state.ctx, 4096, nullptr, nullptr, 0);
    if (!g_state.pd || !g_state.cq) {
        fprintf(stderr, "efa_dp_runtime: PD/CQ creation failed\n");
        return -1;
    }

    ibv_query_gid(g_state.ctx, 1, 0, &g_state.gid);

    // 3. Create QPs for all peers
    int total_qps = (num_ranks - 1) * num_qps_per_peer;
    // For self, we don't need QPs (local copy path)
    // But to keep indexing simple, create placeholder entries
    total_qps = num_ranks * num_qps_per_peer;

    g_state.ibv_qps.resize(total_qps, nullptr);
    g_state.gpu_qps.resize(total_qps, nullptr);
    g_state.sq_attrs.resize(total_qps);
    g_state.rq_attrs.resize(total_qps);

    for (int peer = 0; peer < num_ranks; peer++) {
        if (peer == rank) continue; // Skip self

        for (int qp_id = 0; qp_id < num_qps_per_peer; qp_id++) {
            int idx = peer * num_qps_per_peer + qp_id;
            struct ibv_qp *qp = nullptr;

            if (create_srd_qp(&qp, 0x11111111) != 0) {
                fprintf(stderr, "efa_dp_runtime: QP creation failed for peer=%d qp=%d\n",
                        peer, qp_id);
                return -1;
            }
            g_state.ibv_qps[idx] = qp;

            efa_cuda_qp *gpu_qp = nullptr;
            if (register_qp_for_gpu(qp, &gpu_qp,
                                    &g_state.sq_attrs[idx],
                                    &g_state.rq_attrs[idx]) != 0) {
                fprintf(stderr, "efa_dp_runtime: GPU registration failed for peer=%d qp=%d\n",
                        peer, qp_id);
                return -1;
            }
            g_state.gpu_qps[idx] = gpu_qp;
        }
    }

    fprintf(stderr, "[efa_dp_runtime] Created %d QPs (%d per peer × %d peers)\n",
            total_qps - num_qps_per_peer, num_qps_per_peer, num_ranks - 1);

    // 4. Initialize host-side state
    memset(&g_state.host_state, 0, sizeof(g_state.host_state));
    g_state.host_state.num_peers = num_ranks;
    g_state.host_state.num_qps_per_peer = num_qps_per_peer;
    g_state.host_state.local_rank = rank;
    g_state.host_state.num_ranks = num_ranks;
    g_state.host_state.num_rc_per_pe = num_qps_per_peer;
    g_state.host_state.num_devices_initialized = 1;
    g_state.host_state.gpu_cq = nullptr; // CQ GPU access needs dmabuf

    // Copy GPU QP pointers into device state
    for (int peer = 0; peer < num_ranks; peer++) {
        for (int qp_id = 0; qp_id < num_qps_per_peer; qp_id++) {
            int idx = peer * num_qps_per_peer + qp_id;
            if (idx < EFA_DP_MAX_TOTAL_QPS) {
                g_state.host_state.qps[idx].gpu_qp = g_state.gpu_qps[idx];
            }
        }
    }

    g_state.initialized = true;
    fprintf(stderr, "[efa_dp_runtime] Initialization complete\n");
    return 0;
}

void* alloc(size_t size, size_t alignment) {
    EP_HOST_ASSERT(g_state.initialized);

    // Allocate GPU memory
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess || !ptr) {
        fprintf(stderr, "efa_dp_runtime: cudaMalloc(%zu) failed\n", size);
        return nullptr;
    }

    // Register with EFA for RDMA access
    struct ibv_mr *mr = ibv_reg_mr(g_state.pd, ptr, size,
                                   IBV_ACCESS_LOCAL_WRITE |
                                   IBV_ACCESS_REMOTE_WRITE |
                                   IBV_ACCESS_REMOTE_READ);
    if (!mr) {
        fprintf(stderr, "efa_dp_runtime: ibv_reg_mr failed: %s\n", strerror(errno));
        cudaFree(ptr);
        return nullptr;
    }

    // Store for later cleanup and state setup
    g_state.main_mr = mr;
    g_state.main_buf = ptr;
    g_state.main_buf_size = size;

    // Update device state
    g_state.host_state.local_heap_base = (uint64_t)ptr;
    g_state.host_state.local_lkey = mr->lkey;

    // Update lkey in all QP infos
    for (int i = 0; i < g_state.num_ranks * g_state.num_qps_per_peer; i++) {
        if (i < EFA_DP_MAX_TOTAL_QPS) {
            g_state.host_state.qps[i].lkey = mr->lkey;
        }
    }

    fprintf(stderr, "[efa_dp_runtime] Allocated %zu bytes at %p, lkey=0x%x, rkey=0x%x\n",
            size, ptr, mr->lkey, mr->rkey);
    return ptr;
}

void free(void* ptr) {
    if (g_state.main_mr && g_state.main_buf == ptr) {
        ibv_dereg_mr(g_state.main_mr);
        g_state.main_mr = nullptr;
    }
    cudaFree(ptr);
}

void exchange_memory_info(void* local_ptr, uint32_t local_rkey) {
    // This should be called via OOB (torch.distributed all_gather)
    // For now, store local info and expect peers to call exchange_qp_info()
    // The actual exchange happens through the Python layer using
    // torch.distributed.all_gather

    // Update local peer info
    int rank = g_state.rank;
    g_state.host_state.peers[rank].heap_base = (uint64_t)local_ptr;
    g_state.host_state.peers[rank].rkey = local_rkey;
}

void barrier() {
    // Should use OOB barrier (torch.distributed.barrier)
    // In standalone mode, use simple TCP barrier
    // This is called from Python, so it's a placeholder
}

void finalize() {
    if (!g_state.initialized) return;

    // Destroy GPU QPs
    for (auto gpu_qp : g_state.gpu_qps) {
        if (gpu_qp) efa_cuda_destroy_qp(gpu_qp);
    }

    // Unregister and destroy ibv QPs
    for (int i = 0; i < (int)g_state.ibv_qps.size(); i++) {
        if (g_state.ibv_qps[i]) {
            // Unregister SQ/doorbell from GPU
            if (g_state.sq_attrs[i].buffer) {
                cudaHostUnregister(g_state.sq_attrs[i].buffer);
                void *db_page = (void*)((uintptr_t)g_state.sq_attrs[i].doorbell & ~0xFFF);
                cudaHostUnregister(db_page);
            }
            ibv_destroy_qp(g_state.ibv_qps[i]);
        }
    }

    // Destroy AHs
    for (auto ah : g_state.ahs) {
        if (ah) ibv_destroy_ah(ah);
    }

    // Destroy MR, CQ, PD
    if (g_state.main_mr) ibv_dereg_mr(g_state.main_mr);
    if (g_state.cq) ibv_destroy_cq(g_state.cq);
    if (g_state.pd) ibv_dealloc_pd(g_state.pd);
    if (g_state.ctx) ibv_close_device(g_state.ctx);

    g_state.initialized = false;
    fprintf(stderr, "[efa_dp_runtime] Finalized\n");
}

/**
 * @brief Copy device state to GPU
 *
 * Called after init() + alloc() + exchange() to make the state
 * available to GPU kernels via the efa_dp_state_d global.
 */
void copy_state_to_gpu() {
    cudaMemcpyToSymbol(efa_dp_state_d, &g_state.host_state,
                       sizeof(efa_dp_device_state));
}

/**
 * @brief Set up Address Handles for all peers
 *
 * Called after exchanging GIDs with all peers via OOB.
 *
 * @param peer_gids  Array of GIDs, one per rank
 * @param peer_qp_nums Array of QP numbers, one per rank
 * @param peer_rkeys   Array of rkeys, one per rank
 * @param peer_addrs   Array of buffer base addresses, one per rank
 */
void setup_peers(const union ibv_gid *peer_gids,
                 const uint32_t *peer_qp_nums,
                 const uint32_t *peer_rkeys,
                 const uint64_t *peer_addrs) {
    EP_HOST_ASSERT(g_state.initialized);

    g_state.ahs.resize(g_state.num_ranks, nullptr);
    g_state.ah_nums.resize(g_state.num_ranks, 0);

    for (int peer = 0; peer < g_state.num_ranks; peer++) {
        if (peer == g_state.rank) continue;

        // Create AH
        struct ibv_ah_attr ah_attr = {};
        ah_attr.is_global = 1;
        ah_attr.grh.dgid = peer_gids[peer];
        ah_attr.port_num = 1;

        struct ibv_ah *ah = ibv_create_ah(g_state.pd, &ah_attr);
        EP_HOST_ASSERT(ah != nullptr);

        struct efadv_ah_attr efa_ah = {};
        efadv_query_ah(ah, &efa_ah, sizeof(efa_ah));

        g_state.ahs[peer] = ah;
        g_state.ah_nums[peer] = efa_ah.ahn;

        // Update device state
        g_state.host_state.peers[peer].heap_base = peer_addrs[peer];
        g_state.host_state.peers[peer].rkey = peer_rkeys[peer];
        g_state.host_state.peers[peer].qp_num = peer_qp_nums[peer];
        g_state.host_state.peers[peer].ah_num = efa_ah.ahn;

        fprintf(stderr, "[efa_dp_runtime] Peer %d: qp=%u rkey=0x%x ah=%u addr=0x%lx\n",
                peer, peer_qp_nums[peer], peer_rkeys[peer], efa_ah.ahn, peer_addrs[peer]);
    }

    // Copy updated state to GPU
    copy_state_to_gpu();
}

} // namespace efa_dp_runtime
} // namespace deep_ep
