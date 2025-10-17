// EFA Device State Definition
// This file provides the actual definition of the global EFA device state variable.
// The declaration is in efa_device_fixed.cuh

#include "efa_device_fixed.cuh"

namespace deep_ep {

// Define the global device state with default values
// This will be initialized by host code via cudaMemcpyToSymbol()
__device__ nvshmemi_efa_device_state_t nvshmemi_efa_device_state_d = {
    .num_rc_per_pe = 1,            // Default: 1 channel per PE
    .num_devices_initialized = 1    // Default: 1 device initialized
};

}  // namespace deep_ep
