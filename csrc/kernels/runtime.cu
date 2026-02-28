#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifdef USE_EFA_DP_DIRECT
// EFA-DP-direct mode: use efa-dp-direct runtime instead of NVSHMEM
#include "efa_dp_direct_runtime.cuh"
#elif !defined(DISABLE_NVSHMEM)
#include "nvshmem.h"
// #include "ibgda_device.cuh"
#include "efa_device.cuh"

// Define the global EFA device state variable outside namespace
// This is the actual definition - declared extern in efa_device.cuh
__device__ deep_ep::nvshmemi_efa_device_state_t nvshmemi_efa_device_state_d;
#endif

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace internode {

#ifdef USE_EFA_DP_DIRECT
// ============================================================================
// EFA-DP-direct backend: GPU-direct RDMA via efa-dp-direct
// ============================================================================

std::vector<uint8_t> get_unique_id() {
    return efa_dp_runtime::get_unique_id();
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    // Determine number of QPs per peer
    // For LL mode: one QP per local expert
    // For normal mode: one QP per channel
    int num_local_experts = 256 / (num_ranks / NUM_MAX_NVL_PEERS); // TODO: get from config
    int num_qps_per_peer = low_latency_mode ? num_local_experts : 1;

    int ret = efa_dp_runtime::init(rank, num_ranks, num_qps_per_peer, low_latency_mode);
    if (ret != 0) {
        fprintf(stderr, "[runtime] efa_dp_runtime::init failed: %d\n", ret);
        return ret;
    }

    // OOB exchange of QP info happens through Python layer
    // using torch.distributed.all_gather()
    return rank;
}

void* alloc(size_t size, size_t alignment) {
    return efa_dp_runtime::alloc(size, alignment);
}

void free(void* ptr) {
    efa_dp_runtime::free(ptr);
}

void barrier() {
    efa_dp_runtime::barrier();
}

void finalize() {
    efa_dp_runtime::finalize();
}

#elif !defined(DISABLE_NVSHMEM)
// ============================================================================
// NVSHMEM backend (original + EFA native API)
// ============================================================================

nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % NUM_MAX_NVL_PEERS,
                                                  NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    return nvshmem_align(alignment, size);
}

void free(void* ptr) {
    nvshmem_free(ptr);
}

void barrier() {
    nvshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}
#endif

}  // namespace internode

}  // namespace deep_ep
