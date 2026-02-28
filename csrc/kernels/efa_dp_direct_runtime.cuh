/**
 * @file efa_dp_direct_runtime.cuh
 * @brief Host-side runtime for efa-dp-direct integration with DeepEP
 *
 * Provides initialization, memory management, and teardown functions
 * that replace NVSHMEM for the EFA-DP-direct GPU-direct RDMA path.
 *
 * Initialization flow:
 *   1. Open EFA device, create PD, CQ
 *   2. For each peer: create SRD QP with RDMA_WRITE, transition to RTS
 *   3. Query hardware pointers via efadv_query_qp_wqs()
 *   4. Register SQ/doorbell with cudaHostRegister(IoMemory)
 *   5. Create efa-dp-direct GPU QP via efa_cuda_create_qp()
 *   6. Exchange QP info + MR info via OOB (torch.distributed)
 *   7. Copy device state to GPU constant memory
 *
 * Memory management:
 *   - alloc(): cudaMalloc + ibv_reg_mr (replaces nvshmem_align)
 *   - free(): ibv_dereg_mr + cudaFree (replaces nvshmem_free)
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace deep_ep {
namespace efa_dp_runtime {

/**
 * @brief Initialize EFA-DP-direct runtime
 *
 * Opens EFA device, creates QPs for all peers, registers memory regions,
 * and sets up GPU-side QP/CQ handles.
 *
 * @param rank           Local rank (PE index)
 * @param num_ranks      Total number of ranks
 * @param num_qps_per_peer Number of QPs per peer (typically num_local_experts)
 * @param low_latency_mode Whether to use LL mode QP mapping
 * @return 0 on success, negative on error
 */
int init(int rank, int num_ranks, int num_qps_per_peer, bool low_latency_mode);

/**
 * @brief Allocate GPU memory and register with EFA for RDMA access
 *
 * Replaces nvshmem_align(). Allocates CUDA device memory and registers
 * it with the EFA PD for remote RDMA access.
 *
 * @param size      Number of bytes to allocate
 * @param alignment Memory alignment (minimum 128 bytes)
 * @return Pointer to allocated GPU memory, or nullptr on failure
 */
void* alloc(size_t size, size_t alignment);

/**
 * @brief Free GPU memory and deregister from EFA
 * @param ptr Pointer previously returned by alloc()
 */
void free(void* ptr);

/**
 * @brief Global barrier across all ranks
 *
 * Uses OOB (out-of-band) communication for synchronization.
 * This is needed for initialization and cleanup phases.
 */
void barrier();

/**
 * @brief Exchange buffer addresses and rkeys with all peers
 *
 * After alloc(), call this to distribute memory info to all peers
 * so they can construct RDMA Write operations.
 *
 * @param local_ptr   Local buffer pointer
 * @param local_rkey  Local MR rkey
 */
void exchange_memory_info(void* local_ptr, uint32_t local_rkey);

/**
 * @brief Get a unique ID for bootstrapping (replaces nvshmemx_get_uniqueid)
 * @return Serialized unique ID bytes
 */
std::vector<uint8_t> get_unique_id();

/**
 * @brief Finalize and clean up all EFA resources
 */
void finalize();

} // namespace efa_dp_runtime
} // namespace deep_ep
