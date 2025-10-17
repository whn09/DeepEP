// EFA device operations for DeepEP
// Uses standard NVSHMEM API for Amazon EFA network support
// This is a simplified implementation that relies on NVSHMEM's EFA transport layer
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include <nvshmem.h>

namespace deep_ep {

// ============================================================================
// State Management - Simplified for EFA
// ============================================================================

// Simplified device state structure for EFA
// Unlike IBGDA which needs detailed QP management, EFA uses NVSHMEM's abstraction
typedef struct {
    int num_rc_per_pe;              // Number of communication channels per PE
    int num_devices_initialized;    // Number of devices (GPUs) initialized
} nvshmemi_efa_device_state_t;

// Global device state (should be initialized by host code)
// For EFA, these values typically match the number of channels/devices configured
// Declaration only - definition is in efa_device_state.cu
extern __device__ nvshmemi_efa_device_state_t nvshmemi_efa_device_state_d;

// Get EFA device state
// This function provides compatibility with IBGDA code that calls ibgda_get_state()
__device__ static __forceinline__ nvshmemi_efa_device_state_t* ibgda_get_state() {
    return &nvshmemi_efa_device_state_d;
}

// ============================================================================
// P2P Pointer Translation
// ============================================================================

// Get P2P pointer for direct NVLink access (if available)
// Returns 0 if P2P is not available (need to use RDMA)
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank - no translation needed
    if (rank == dst_rank)
        return ptr;

    // Try to get P2P pointer using nvshmem_ptr
    // This will return NULL if P2P (NVLink) is not available
    void* remote_ptr = nvshmem_ptr(reinterpret_cast<void*>(ptr), dst_rank);

    // Return 0 to indicate RDMA should be used instead
    if (remote_ptr == NULL)
        return 0;

    return reinterpret_cast<uint64_t>(remote_ptr);
}

// ============================================================================
// Warp-level PUT Operation
// ============================================================================

// Warp-level non-blocking PUT operation
// This version uses standard NVSHMEM API and distributes work across the warp
template <bool kAlwaysDoPostSend = false>
__device__ __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
                           int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Check for P2P capability first (NVLink direct access)
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(req_rptr, nvshmem_my_pe(), dst_pe);

    if (p2p_ptr != 0) {
        // Use direct NVLink P2P copy
        if (lane_id == 0) {
            memcpy(reinterpret_cast<void*>(p2p_ptr),
                   reinterpret_cast<const void*>(req_lptr),
                   bytes);
        }
        __syncwarp();
        return;
    }

    // For EFA/RDMA path, use NVSHMEM putmem
    // Only lane 0 performs the operation to avoid duplicates
    if (lane_id == 0) {
        nvshmem_putmem_nbi(reinterpret_cast<void*>(req_rptr),
                          reinterpret_cast<const void*>(req_lptr),
                          bytes,
                          dst_pe);
    }

    // Synchronize warp
    __syncwarp();

    // Optional fence for message ordering
    if (kAlwaysDoPostSend && lane_id == 0) {
        nvshmem_fence();
    }
}

// ============================================================================
// Atomic Operations
// ============================================================================

// Atomic add operation (non-fetching)
__device__ __forceinline__ void
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id,
                               bool is_local_copy = false) {
    if (is_local_copy) {
        // Local atomic operation
        atomicAdd(static_cast<int*>(rptr), value);
        return;
    }

    // Check for P2P capability
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rptr),
                                           nvshmem_my_pe(), pe);

    if (p2p_ptr != 0) {
        // Use direct NVLink P2P atomic
        atomicAdd(reinterpret_cast<int*>(p2p_ptr), value);
        return;
    }

    // Use NVSHMEM atomic for EFA/RDMA path
    nvshmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
}

// ============================================================================
// RMA Put Operation (single int value)
// ============================================================================

// RMA Put operation for single int value
// This is used in barrier synchronization
__device__ __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int qp_id,
    uint32_t imm = std::numeric_limits<uint32_t>::max()) {

    // Check for P2P capability
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rptr),
                                           nvshmem_my_pe(), dst_pe);

    if (p2p_ptr != 0) {
        // Use direct NVLink P2P write
        *reinterpret_cast<int*>(p2p_ptr) = value;
        __threadfence_system();  // Ensure visibility
        return;
    }

    // Use NVSHMEM put for EFA/RDMA path
    // nvshmem_int_p is non-blocking and efficient for single int
    nvshmem_int_p(rptr, value, dst_pe);

    // Note: Caller must use nvshmem_quiet() to ensure completion
}

// ============================================================================
// Synchronization Operations
// ============================================================================

// Quiet operation - wait for all outstanding operations to a PE to complete
__device__ static __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    // Use NVSHMEM quiet to ensure all operations are complete
    nvshmem_quiet();
}

// Fence operation - order operations
__device__ static __forceinline__ void
nvshmem_fence_wrapper() {
    nvshmem_fence();
}

// Barrier operation
__device__ static __forceinline__ void
nvshmem_barrier_wrapper() {
    nvshmem_barrier_all();
}

// ============================================================================
// Utility Functions
// ============================================================================

// Check if PE is local (same as calling PE)
__device__ static __forceinline__ bool
is_local_pe(int pe) {
    return pe == nvshmem_my_pe();
}

// Get number of PEs
__device__ static __forceinline__ int
get_n_pes() {
    return nvshmem_n_pes();
}

// Get my PE number
__device__ static __forceinline__ int
get_my_pe() {
    return nvshmem_my_pe();
}

}  // namespace deep_ep
