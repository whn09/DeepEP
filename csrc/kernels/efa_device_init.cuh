// EFA Device Initialization Helper
// Helper functions to initialize EFA device state from host code
#pragma once

#include "efa_device_fixed.cuh"
#include <cuda_runtime.h>

namespace deep_ep {

// Host-side function to initialize EFA device state
// This should be called during setup, typically after nvshmem_init()
//
// Parameters:
//   num_channels: Number of communication channels per PE (typically matches your kernel channel count)
//   num_devices: Number of GPUs initialized per node
inline void init_efa_device_state(int num_channels, int num_devices) {
    nvshmemi_efa_device_state_t host_state;
    host_state.num_rc_per_pe = num_channels;
    host_state.num_devices_initialized = num_devices;

    // Copy to device symbol
    cudaError_t err = cudaMemcpyToSymbol(nvshmemi_efa_device_state_d,
                                         &host_state,
                                         sizeof(nvshmemi_efa_device_state_t));

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize EFA device state: %s\n",
                cudaGetErrorString(err));
    }
}

// Helper to get recommended channel count based on system configuration
inline int get_recommended_channel_count() {
    // For EFA, typically use 1-2 channels per PE
    // This can be tuned based on your workload
    return 1;
}

}  // namespace deep_ep
