/**
 * @file
 * @brief EFA (Elastic Fabric Adapter) protocol implementation for NVSHMEM
 * 
 * This module provides EC2 EFA protocol integration with NVSHMEM native API.
 * Portions modified from the original DeepEP implementation to support
 * AWS EFA network fabric for high-performance inter-GPU communication.
 * 
 * @note Based on DeepEP implementation with EFA-specific optimizations
 */
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include <nvshmem.h>

// Non-existent header file removed from original implementation
// #include <nvshmemx.h>

namespace deep_ep {

/**
 * @brief Define missing NVSHMEM constant for maximum processing elements
 * 
 * This constant defines the upper limit for the number of processing elements
 * (PEs) that can participate in NVSHMEM operations. Only defined if not
 * already present in the NVSHMEM headers.
 */
#ifndef NVSHMEM_MAX_NUM_PES
#define NVSHMEM_MAX_NUM_PES 1024
#endif

/**
 * @brief Maximum inline data size for EFA operations
 * 
 * EFA supports inline data up to 64 bytes, which can be sent without
 * requiring additional memory registration or buffer management.
 */
constexpr int EFA_MAX_INLINE_SIZE = 64;

/**
 * @brief Maximum scatter-gather elements for EFA
 * 
 * Defines the maximum number of scatter-gather list entries that can
 * be used in a single EFA operation for vectorized I/O.
 */
constexpr int EFA_MAX_SGE = 32;

/**
 * @brief Simplified request structure for EFA operations
 * 
 * Custom request structure that replaces nvshmemx_request_t from the
 * extended NVSHMEM API, providing a lightweight alternative for tracking
 * asynchronous EFA operations.
 */
typedef struct {
    void *addr;        // Base address of the operation
    size_t size;       // Size of data in bytes
    int pe;            // Target processing element ID
    int status;        // Current status of the request
} efa_request_t;

/**
 * @brief EFA device state structure
 *
 * Maintains the global state for EFA device operations, including device
 * context, PE information, and EFA-specific memory registration keys.
 * This structure is shared across all CUDA kernels on a device.
 */
typedef struct {
    void *efa_device_context;              // EFA device context handle
    uint32_t num_devices;                   // Total number of EFA devices
    uint32_t local_device_id;              // Local device ID
    uint32_t num_pes;                      // Number of processing elements
    bool use_async_operations;             // Enable asynchronous operation mode

    // Compatibility fields for IBGDA-style queue pair management
    // For EFA, we use a simplified model where QPs are managed by NVSHMEM
    uint32_t num_rc_per_pe;                // Number of reliable connections per PE (typically 1 for EFA)
    uint32_t num_devices_initialized;      // Number of initialized devices (typically 1)

    // EFA-specific memory registration keys for RDMA operations
    struct {
        uint32_t *lkeys;                   // Local memory keys array
        uint32_t *rkeys;                   // Remote memory keys array
    } efa_keys;
} nvshmemi_efa_device_state_t;

/**
 * @brief EFA queue pair structure
 * 
 * Represents a queue pair (QP) for EFA communication with a remote PE.
 * Each QP manages send/receive operations and maintains a pool of requests
 * for tracking asynchronous operations.
 */
typedef struct {
    uint32_t qp_id;                        // Queue pair identifier
    uint32_t dev_idx;                      // Device index
    uint32_t remote_pe;                    // Remote processing element ID
    efa_request_t *request_pool;           // Pool of custom request structures
    uint32_t next_request_idx;             // Index of next available request
    uint32_t max_requests;                 // Maximum requests in pool
    
    // EFA-specific queue pair management fields
    void *efa_qp_handle;                   // EFA queue pair handle
    uint32_t send_credits;                 // Available send credits
    uint32_t max_send_credits;             // Maximum send credits
} nvshmemi_efa_device_qp_t;

} // namespace deep_ep

// Declare global EFA device state outside namespace to match linker expectations
// This variable must be declared at global scope to be accessible across translation units
// The actual definition is in runtime.cu
extern __device__ deep_ep::nvshmemi_efa_device_state_t nvshmemi_efa_device_state_d;

namespace deep_ep {

/**
 * @brief Forward declaration of P2P pointer translation function
 * 
 * Translates a local heap pointer to a peer-accessible pointer for
 * NVLink P2P operations. Full implementation follows later in this file.
 * 
 * @param ptr Local pointer to translate
 * @param rank Source rank ID
 * @param dst_rank Destination rank ID
 * @return Translated P2P pointer, or 0 if P2P is not available
 */
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank);

/**
 * @brief Retrieve the global EFA device state
 *
 * @return Pointer to the global EFA device state structure
 */
__device__ static __forceinline__
nvshmemi_efa_device_state_t* efa_get_state() {
    return &::nvshmemi_efa_device_state_d;
}

/**
 * @brief Retrieve the global EFA device state (IBGDA compatibility alias)
 *
 * @return Pointer to the global EFA device state structure
 */
__device__ static __forceinline__
nvshmemi_efa_device_state_t* ibgda_get_state() {
    return &::nvshmemi_efa_device_state_d;
}

/**
 * @brief Retrieve an EFA queue pair for a specific PE and ID
 * 
 * Returns a pointer to a queue pair from a statically allocated pool.
 * Uses fixed-size arrays to avoid dynamic allocation in device code,
 * which improves performance and avoids memory management complexity.
 * 
 * @param pe Processing element ID
 * @param id Queue pair ID
 * @return Pointer to the requested queue pair structure
 * @note Performs boundary checking with modulo arithmetic for safety
 */
__device__ static __forceinline__
nvshmemi_efa_device_qp_t* efa_get_qp(int pe, int id) {
    // Use fixed-size array to avoid dynamic memory allocation in device code
    constexpr int MAX_PES = 1024;
    constexpr int QPS_PER_PE = 8;
    static __device__ nvshmemi_efa_device_qp_t qp_pool[MAX_PES][QPS_PER_PE];
    
    // Boundary checking using modulo to ensure valid array access
    int safe_pe = pe % MAX_PES;
    int safe_id = id % QPS_PER_PE;
    return &qp_pool[safe_pe][safe_id];
}

/**
 * @brief Convert local symmetric heap pointer to remote heap pointer
 * 
 * Translates a local heap pointer to the corresponding address in a remote
 * PE's symmetric heap by calculating the offset and applying it to the
 * remote heap base address.
 * 
 * @param local_ptr Local symmetric heap pointer
 * @param dst_pe Destination processing element ID
 * @return Remote symmetric heap pointer for the specified PE
 */
__device__ static __forceinline__
uint64_t efa_get_symmetric_heap_ptr(uint64_t local_ptr, int dst_pe) {
    // Calculate offset from local heap base
    auto heap_base = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto offset = local_ptr - heap_base;
    // Apply offset to remote heap base to get remote address
    return reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + offset;
}

/**
 * @brief Check if the specified PE is the local PE
 * 
 * @param pe Processing element ID to check
 * @return true if pe is the local PE, false otherwise
 */
__device__ static __forceinline__
bool efa_is_local_pe(int pe) {
    return pe == nvshmemi_device_state_d.mype;
}

/**
 * @brief Warp-level non-blocking PUT operation via EFA/NVSHMEM
 * 
 * Performs a distributed PUT operation where warp lanes collaborate to
 * transfer data from local to remote memory. Automatically selects the
 * optimal transfer method: local memcpy, NVLink P2P, or EFA/NVSHMEM RDMA.
 * 
 * The operation divides large transfers among warp lanes for improved
 * bandwidth utilization. Each lane handles a chunk of the total transfer.
 * 
 * @tparam kAlwaysDoPostSend If true, always perform fence after operation
 * @param req_rptr Remote destination pointer
 * @param req_lptr Local source pointer
 * @param bytes Number of bytes to transfer
 * @param dst_pe Destination processing element ID
 * @param qp_id Queue pair ID to use for the operation
 * @param lane_id Lane ID within the warp (0-31)
 * @param message_idx Message index for tracking
 * 
 * @note All threads in the warp must call this function together
 * @warning Assumes warp-synchronous execution model
 */
template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // Fast path: check if this is a local copy operation
    if (efa_is_local_pe(dst_pe)) {
        if (lane_id == 0) {
            // Perform local memory copy using standard memcpy
            memcpy(reinterpret_cast<void*>(req_rptr), 
                   reinterpret_cast<const void*>(req_lptr), 
                   bytes);
        }
        __syncwarp();
        return;
    }

    // Check for NVLink P2P capability between source and destination
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(req_rptr, nvshmemi_device_state_d.mype, dst_pe);
    if (p2p_ptr != 0) {
        if (lane_id == 0) {
            // Direct NVLink P2P copy for high-bandwidth intra-node transfers
            memcpy(reinterpret_cast<void*>(p2p_ptr), 
                   reinterpret_cast<const void*>(req_lptr), 
                   bytes);
        }
        __syncwarp();
        return;
    }

    // Fall back to EFA/NVSHMEM for remote inter-node communication
    auto qp = efa_get_qp(dst_pe, qp_id);
    
    // Calculate symmetric heap address on remote PE
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(req_rptr, dst_pe);
    
    // Distribute work among warp lanes for large transfers (32 lanes)
    const size_t chunk_size = (bytes + 31) / 32;
    size_t my_offset = lane_id * chunk_size;
    size_t my_size = min(chunk_size, bytes - my_offset);
    
    if (my_size > 0 && my_offset < bytes) {
        // Each lane performs NVSHMEM put operation for its chunk
        if (my_size <= 8) {
            // Use optimized typed put operations for small data sizes
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
            // Use bulk transfer for larger chunks to amortize overhead
            nvshmem_putmem(reinterpret_cast<void*>(remote_ptr + my_offset),
                          reinterpret_cast<const void*>(req_lptr + my_offset),
                          my_size, dst_pe);
        }
    }

    // Synchronize all lanes in warp before returning
    __syncwarp();
    
    // Optional: Add completion signaling for the last message in a batch
    if (kAlwaysDoPostSend && lane_id == 0) {
        // Ensure all operations are visible to remote PEs
        nvshmem_fence();
    }
}

/**
 * @brief Non-fetching atomic ADD operation via EFA/NVSHMEM
 * 
 * Performs an atomic addition on remote memory without fetching the result.
 * Automatically selects optimal path: local atomic, NVLink P2P atomic, or
 * EFA/NVSHMEM remote atomic operation.
 * 
 * @param rptr Remote pointer to the integer to modify
 * @param value Value to add to the remote integer
 * @param pe Target processing element ID
 * @param qp_id Queue pair ID (used for routing)
 * @param is_local_copy Force local copy path if true
 * 
 * @note Non-fetching atomics have lower latency than fetch-and-add
 */
__device__ __forceinline__ void 
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    // Handle local or forced-local case with standard CUDA atomic
    if (is_local_copy || efa_is_local_pe(pe)) {
        atomicAdd(static_cast<int*>(rptr), value);
        return;
    }

    // Check for NVLink P2P capability for intra-node atomics
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rptr), 
                                           nvshmemi_device_state_d.mype, pe);
    if (p2p_ptr != 0) {
        // Direct NVLink P2P atomic operation
        atomicAdd(reinterpret_cast<int*>(p2p_ptr), value);
        return;
    }

    // Use NVSHMEM atomic add for remote EFA communication
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(reinterpret_cast<uint64_t>(rptr), pe);
    
    // Perform remote atomic add via NVSHMEM
    nvshmem_int_atomic_add(reinterpret_cast<int*>(remote_ptr), value, pe);
}

/**
 * @brief Template helper for type-safe atomic add operations
 * 
 * Provides compile-time dispatch to the appropriate NVSHMEM atomic operation
 * based on the data type. Supports all standard integer types including
 * signed, unsigned, and long long variants.
 * 
 * @tparam T Data type for the atomic operation
 * @param rptr Remote pointer to modify
 * @param value Value to add
 * @param pe Target processing element ID
 * 
 * @note Uses C++17 if constexpr for zero-overhead type dispatch
 * @warning Falls back to int atomic for unsupported types
 */
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
        // Fallback for unsupported types - reinterpret as int
        static_assert(sizeof(T) <= 8, "Unsupported atomic type size");
        nvshmem_int_atomic_add(reinterpret_cast<int*>(remote_ptr), 
                              *reinterpret_cast<const int*>(&value), pe);
    }
}

/**
 * @brief Wait for all outstanding operations to a PE to complete
 * 
 * Ensures that all previously initiated operations to the specified PE
 * have completed and are visible to the remote PE. Uses NVSHMEM fence
 * and quiet operations for completion semantics.
 * 
 * @param dst_pe Destination processing element to wait for
 * @param qp_id Queue pair ID (used for tracking)
 * 
 * @note This is a blocking operation that may have significant latency
 */
__device__ static __forceinline__ void
nvshmemi_efa_quiet(int dst_pe, int qp_id) {
    // Use NVSHMEM fence to ensure operation ordering
    nvshmem_fence();
    
    // Wait for all operations to complete (PE-specific quiet)
    nvshmem_quiet();
}

/**
 * @brief Memory fence for EFA operations
 * 
 * Ensures that all previously initiated NVSHMEM operations are ordered
 * before any subsequent operations. Does not wait for completion.
 */
__device__ static __forceinline__ void
nvshmemi_efa_fence() {
    nvshmem_fence();
}

/**
 * @brief Global barrier across all processing elements
 * 
 * Synchronizes all PEs in the NVSHMEM job. All PEs must call this
 * function before any can proceed past the barrier.
 * 
 * @warning This is a collective operation - all PEs must participate
 */
__device__ static __forceinline__ void
nvshmemi_efa_barrier_all() {
    nvshmem_barrier_all();
}

/**
 * @brief Translate local pointer to P2P-accessible pointer
 * 
 * Converts a local symmetric heap pointer to a peer-accessible pointer
 * for NVLink P2P direct memory access. Returns 0 if P2P is not available
 * between the specified ranks.
 * 
 * This function enables high-bandwidth, low-latency intra-node communication
 * when GPUs are connected via NVLink.
 * 
 * @param ptr Local heap pointer to translate
 * @param rank Source rank ID
 * @param dst_rank Destination rank ID
 * @return P2P-accessible pointer, or 0 if P2P is not available
 * 
 * @note Returns original pointer if rank equals dst_rank (local access)
 */
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank requires no address translation
    if (rank == dst_rank)
        return ptr;
    
    // Check if P2P base address array is valid
    if (nvshmemi_device_state_d.peer_heap_base_p2p == nullptr)
        return 0;
        
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // Check if RDMA connection is established (peer_base == 0 means not connected)
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled - calculate remote address
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

/**
 * @brief Initialize EFA device state
 *
 * Sets up the EFA device state structure with values from the global
 * NVSHMEM device state. Should be called once during initialization.
 */
__device__ static __forceinline__ void
efa_init_device_state() {
    auto state = efa_get_state();
    state->num_pes = nvshmemi_device_state_d.npes;
    state->use_async_operations = true;

    // Initialize compatibility fields for IBGDA-style code
    // EFA uses NVSHMEM's internal connection management, so we use simple defaults
    state->num_rc_per_pe = 1;              // One logical connection per PE
    state->num_devices_initialized = 1;    // Single device per process
}

/**
 * @brief Check if EFA subsystem is ready for operations
 * 
 * Verifies that the EFA subsystem has been properly initialized by
 * checking if the symmetric heap has been allocated.
 * 
 * @return true if EFA is ready, false otherwise
 */
__device__ static __forceinline__ bool
efa_is_ready() {
    // Heap base pointer is a good indicator of initialization status
    return nvshmemi_device_state_d.heap_base != nullptr;
}

/**
 * @brief RMA Put operation for single int value
 *
 * Performs a remote memory write of a single integer value to the specified
 * processing element. This operation is commonly used in barrier synchronization
 * and counter updates. The function automatically selects the optimal path:
 * - NVLink P2P direct write for intra-node communication
 * - NVSHMEM int_p for inter-node EFA/RDMA communication
 *
 * @param rptr Remote pointer to the integer location to write
 * @param value The integer value to write
 * @param dst_pe Destination processing element ID
 * @param qp_id Queue pair ID (used for routing/load balancing)
 * @param imm Immediate data (optional, for future extensibility)
 *
 * @note This is a non-blocking operation. Use nvshmem_quiet() or
 *       nvshmemi_ibgda_quiet() to ensure completion if ordering is required.
 * @note The operation uses system-wide memory fence for P2P writes to ensure
 *       visibility across all processing elements.
 */
__device__ __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int qp_id,
    uint32_t imm = std::numeric_limits<uint32_t>::max()) {

    // Fast path: check if this is a local write operation
    if (efa_is_local_pe(dst_pe)) {
        *rptr = value;
        __threadfence_system();
        return;
    }

    // Check for NVLink P2P capability for intra-node writes
    uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rptr),
                                           nvshmemi_device_state_d.mype, dst_pe);

    if (p2p_ptr != 0) {
        // Direct NVLink P2P write with system-wide fence
        *reinterpret_cast<int*>(p2p_ptr) = value;
        __threadfence_system();  // Ensure visibility across all PEs
        return;
    }

    // Fall back to NVSHMEM put for remote inter-node EFA/RDMA communication
    // Calculate symmetric heap address on remote PE
    uint64_t remote_ptr = efa_get_symmetric_heap_ptr(reinterpret_cast<uint64_t>(rptr), dst_pe);

    // Use nvshmem_int_p for efficient single integer write
    // This is a non-blocking operation optimized for small data transfers
    nvshmem_int_p(reinterpret_cast<int*>(remote_ptr), value, dst_pe);

    // Note: Caller should use nvshmem_quiet() if completion guarantee is needed
}

/**
 * @brief Backward-compatible alias for nvshmemi_efa_quiet
 *
 * Provides compatibility with legacy code that uses the ibgda naming
 * convention. Simply forwards to the EFA implementation.
 *
 * @param dst_pe Destination processing element ID
 * @param qp_id Queue pair ID
 */
__device__ static __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    nvshmemi_efa_quiet(dst_pe, qp_id);
}

} // namespace deep_ep