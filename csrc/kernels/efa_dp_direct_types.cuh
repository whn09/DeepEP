/**
 * @file efa_dp_direct_types.cuh
 * @brief Shared type definitions for efa-dp-direct integration.
 *
 * This header contains only struct/constant definitions used by both
 * the host runtime (efa_dp_direct_runtime.cu) and device code
 * (efa_dp_direct_device.cuh). It does NOT include any __device__
 * function definitions that could cause multiple-definition errors.
 */
#pragma once

#include "efa_cuda_dp.h"

namespace deep_ep {

// Maximum number of RDMA peers supported
#define EFA_DP_MAX_PEERS 128
// Maximum number of QPs per peer (one per local expert / channel)
#define EFA_DP_MAX_QPS_PER_PEER 64
// Maximum total QPs
#define EFA_DP_MAX_TOTAL_QPS (EFA_DP_MAX_PEERS * EFA_DP_MAX_QPS_PER_PEER)

/**
 * Per-peer remote memory info for address translation.
 */
struct efa_dp_remote_info {
    uint64_t heap_base;
    uint32_t rkey;
    uint32_t qp_num;
    uint16_t ah_num;
    uint16_t reserved;
};

/**
 * Per-QP info: maps (peer, qp_id) to an efa_cuda_qp device handle.
 */
struct efa_dp_qp_info {
    efa_cuda_qp *gpu_qp;
    uint32_t lkey;
};

/**
 * Global device state for efa-dp-direct operations.
 * Initialized by host-side runtime and copied to device memory.
 */
struct efa_dp_device_state {
    efa_dp_remote_info peers[EFA_DP_MAX_PEERS];
    int num_peers;

    efa_dp_qp_info qps[EFA_DP_MAX_TOTAL_QPS];
    int num_qps_per_peer;

    efa_cuda_cq *gpu_cq;

    uint64_t local_heap_base;
    uint32_t local_lkey;
    int local_rank;
    int num_ranks;

    uint64_t p2p_bases[EFA_DP_MAX_PEERS];

    uint32_t num_rc_per_pe;
    uint32_t num_devices_initialized;
};

} // namespace deep_ep
