#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace deep_ep {

// 替代ibgda_get_p2p_ptr函数
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // 本地rank，不需要映射
    if (rank == dst_rank)
        return ptr;
    
    // 使用nvshmem_ptr获取远程指针
    void* remote_ptr = nvshmem_ptr(reinterpret_cast<void*>(ptr), dst_rank);
    if (remote_ptr == NULL)
        return 0;
    
    return reinterpret_cast<uint64_t>(remote_ptr);
}

// 替代nvshmemi_ibgda_put_nbi_warp函数
template <bool kAlwaysDoPostSend = false>
__device__ __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // 使用nvshmemx_uint64_put_nbi_warp替代
    nvshmemx_uint64_put_nbi_warp(
        reinterpret_cast<uint64_t*>(req_rptr),
        reinterpret_cast<const uint64_t*>(req_lptr),
        bytes / sizeof(uint64_t),
        dst_pe
    );
}

// 替代nvshmemi_ibgda_amo_nonfetch_add函数
__device__ __forceinline__ void 
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned int*>(rptr), value);
    } else {
        // 使用nvshmem_int_atomic_add替代
        nvshmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
    }
}

// 替代nvshmemi_ibgda_quiet函数
__device__ __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    // 使用nvshmem_quiet确保所有操作完成
    nvshmem_quiet();
}

// 替代translate_dst_rdma_rank函数的实现
template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

// // 替代nvshmem_sync_with_same_gpu_idx函数的实现
// template <bool kLowLatencyMode>
// __forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
//     kLowLatencyMode ? nvshmem_sync(rdma_team) : nvshmem_sync_all();
// }

} // namespace deep_ep