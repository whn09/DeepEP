// Portions modified to use EC2 EFA protocol with NVSHMEM native API
// Based on DeepEP implementation
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include <nvshmem.h>
// 移除不存在的头文件
// #include <nvshmemx.h>

namespace deep_ep {

// 定义缺失的常量
#ifndef NVSHMEM_MAX_NUM_PES
#define NVSHMEM_MAX_NUM_PES 1024
#endif

// EFA-specific constants
constexpr int EFA_MAX_INLINE_SIZE = 64;
constexpr int EFA_MAX_SGE = 32;

// 简化的请求结构，替代 nvshmemx_request_t
typedef struct {
    void *addr;
    size_t size;
    int pe;
    int status;
} efa_request_t;

// EFA device state structure
typedef struct {
    void *efa_device_context;
    uint32_t num_devices;
    uint32_t local_device_id;
    uint32_t num_pes;
    bool use_async_operations;
    // EFA-specific fields
    struct {
        uint32_t *lkeys;
        uint32_t *rkeys;
    } efa_keys;
} nvshmemi_efa_device_state_t;

// EFA queue pair structure
typedef struct {
    uint32_t qp_id;
    uint32_t dev_idx;
    uint32_t remote_pe;
    efa_request_t *request_pool;  // 使用自定义的请求结构
    uint32_t next_request_idx;
    uint32_t max_requests;
    // EFA-specific fields
    void *efa_qp_handle;
    uint32_t send_credits;
    uint32_t max_send_credits;
} nvshmemi_efa_device_qp_t;

// Global EFA device state
extern __device__ nvshmemi_efa_device_state_t nvshmemi_efa_device_state_d;

// 前向声明 nvshmemi_get_p2p_ptr 函数
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank);

__device__ static __forceinline__
nvshmemi_efa_device_state_t* efa_get_state() {
    return &nvshmemi_efa_device_state_d;
}

__device__ static __forceinline__
nvshmemi_efa_device_qp_t* efa_get_qp(int pe, int id) {
    // 使用固定大小的数组，避免动态分配
    constexpr int MAX_PES = 1024;
    constexpr int QPS_PER_PE = 8;
    static __device__ nvshmemi_efa_device_qp_t qp_pool[MAX_PES][QPS_PER_PE];
    
    // 边界检查
    int safe_pe = pe % MAX_PES;
    int safe_id = id % QPS_PER_PE;
    return &qp_pool[safe_pe][safe_id];
}

__device__ static __forceinline__
uint64_t efa_get_symmetric_heap_ptr(uint64_t local_ptr, int dst_pe) {
    // Convert local heap pointer to remote symmetric heap pointer
    auto heap_base = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto offset = local_ptr - heap_base;
    return reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + offset;
}

__device__ static __forceinline__
bool efa_is_local_pe(int pe) {
    return pe == nvshmemi_device_state_d.mype;
}

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Check if this is a local copy
    if (efa_is_local_pe(dst_pe)) {
        if (lane_id == 0) {
            // Local memory copy
            memcpy(reinterpret_cast<void*>(req_rptr), 
                   reinterpret_cast<const void*>(req_lptr), 
                   bytes);
        }
        __syncwarp();
        return;
    }

    // Check for NVLink P2P capability
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(req_rptr, nvshmemi_device_state_d.mype, dst_pe);
    if (p2p_ptr != 0) {
        if (lane_id == 0) {
            // Direct NVLink P2P copy
            memcpy(reinterpret_cast<void*>(p2p_ptr), 
                   reinterpret_cast<const void*>(req_lptr), 
                   bytes);
        }
        __syncwarp();
        return;
    }

    // Use EFA/NVSHMEM for remote communication
    auto qp = efa_get_qp(dst_pe, qp_id);
    
    // Calculate symmetric heap address
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(req_rptr, dst_pe);
    
    // Split work among warp lanes for large transfers
    const size_t chunk_size = (bytes + 31) / 32;  // Divide among 32 lanes
    size_t my_offset = lane_id * chunk_size;
    size_t my_size = min(chunk_size, bytes - my_offset);
    
    if (my_size > 0 && my_offset < bytes) {
        // Use NVSHMEM put operation for each lane's chunk
        if (my_size <= 8) {
            // Use optimized small data put
            if (my_size == 4) {
                nvshmem_int_put(reinterpret_cast<int*>(remote_ptr + my_offset),
                               reinterpret_cast<const int*>(req_lptr + my_offset),
                               1, dst_pe);
            } else if (my_size == 8) {
                nvshmem_long_put(reinterpret_cast<long*>(remote_ptr + my_offset),
                                reinterpret_cast<const long*>(req_lptr + my_offset),
                                1, dst_pe);
            } else {
                nvshmem_putmem(reinterpret_cast<void*>(remote_ptr + my_offset),
                              reinterpret_cast<const void*>(req_lptr + my_offset),
                              my_size, dst_pe);
            }
        } else {
            // Use bulk transfer for larger chunks
            nvshmem_putmem(reinterpret_cast<void*>(remote_ptr + my_offset),
                          reinterpret_cast<const void*>(req_lptr + my_offset),
                          my_size, dst_pe);
        }
    }

    // Ensure all lanes complete before returning
    __syncwarp();
    
    // Optional: Add completion signaling for last message in batch
    if (kAlwaysDoPostSend && lane_id == 0) {
        // Signal completion if needed
        nvshmem_fence();
    }
}

__device__ __forceinline__ void 
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    // Handle local case
    if (is_local_copy || efa_is_local_pe(pe)) {
        atomicAdd(static_cast<int*>(rptr), value);
        return;
    }

    // Check for NVLink P2P capability
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rptr), 
                                           nvshmemi_device_state_d.mype, pe);
    if (p2p_ptr != 0) {
        // Direct NVLink P2P atomic operation
        atomicAdd(reinterpret_cast<int*>(p2p_ptr), value);
        return;
    }

    // Use NVSHMEM atomic add for remote EFA communication
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(reinterpret_cast<uint64_t>(rptr), pe);
    
    // Use NVSHMEM atomic add operation
    nvshmem_int_atomic_add(reinterpret_cast<int*>(remote_ptr), value, pe);
}

// Helper function for atomic operations with different data types
template<typename T>
__device__ __forceinline__ void 
nvshmemi_efa_atomic_add(T *rptr, const T& value, int pe) {
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(reinterpret_cast<uint64_t>(rptr), pe);
    
    if constexpr (std::is_same_v<T, int>) {
        nvshmem_int_atomic_add(reinterpret_cast<int*>(remote_ptr), value, pe);
    } else if constexpr (std::is_same_v<T, long>) {
        nvshmem_long_atomic_add(reinterpret_cast<long*>(remote_ptr), value, pe);
    } else if constexpr (std::is_same_v<T, long long>) {
        nvshmem_longlong_atomic_add(reinterpret_cast<long long*>(remote_ptr), value, pe);
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        nvshmem_uint_atomic_add(reinterpret_cast<unsigned int*>(remote_ptr), value, pe);
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        nvshmem_ulong_atomic_add(reinterpret_cast<unsigned long*>(remote_ptr), value, pe);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        nvshmem_ulonglong_atomic_add(reinterpret_cast<unsigned long long*>(remote_ptr), value, pe);
    } else {
        // Fallback for unsupported types
        static_assert(sizeof(T) <= 8, "Unsupported atomic type size");
        nvshmem_int_atomic_add(reinterpret_cast<int*>(remote_ptr), 
                              *reinterpret_cast<const int*>(&value), pe);
    }
}

// Sync and fence operations for EFA
__device__ static __forceinline__ void
nvshmemi_efa_quiet(int dst_pe, int qp_id) {
    // Use NVSHMEM fence for completion
    nvshmem_fence();
    
    // Optional: PE-specific quiet operation
    nvshmem_quiet();
}

__device__ static __forceinline__ void
nvshmemi_efa_fence() {
    nvshmem_fence();
}

__device__ static __forceinline__ void
nvshmemi_efa_barrier_all() {
    nvshmem_barrier_all();
}

// 实现 nvshmemi_get_p2p_ptr 函数
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank, no need for mapping
    if (rank == dst_rank)
        return ptr;
    
    // 检查是否有有效的P2P基地址数组
    if (nvshmemi_device_state_d.peer_heap_base_p2p == nullptr)
        return 0;
        
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // RDMA connected
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

// Additional EFA-specific utility functions
__device__ static __forceinline__ void
efa_init_device_state() {
    auto state = efa_get_state();
    state->num_pes = nvshmemi_device_state_d.npes;
    state->use_async_operations = true;
}

__device__ static __forceinline__ bool
efa_is_ready() {
    // Check if EFA subsystem is ready
    return nvshmemi_device_state_d.heap_base != nullptr;
}

// 别名函数，保持向后兼容
__device__ static __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    nvshmemi_efa_quiet(dst_pe, qp_id);
}

} // namespace deep_ep