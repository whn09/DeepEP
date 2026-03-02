/**
 * @file efa_dp_direct_device_defs.cu
 * @brief Single compilation unit for efa-dp-direct __device__ function definitions.
 *
 * With -rdc=true, nvlink cannot resolve __device__ symbols from shared libraries.
 * This file includes efa_cuda_dp_impl.cuh ONCE to provide all device function
 * definitions that other TUs reference via declarations from efa_cuda_dp.cuh.
 *
 * Also defines the global __device__ state variable efa_dp_state_d.
 */

#include "efa_cuda_dp.cuh"
#include "efa_cuda_dp_impl.cuh"
#include "efa_dp_direct_types.cuh"

// The global device state — declared extern in efa_dp_direct_device.cuh
namespace deep_ep {
__device__ efa_dp_device_state efa_dp_state_d;
} // namespace deep_ep
