# DeepEP + efa-dp-direct Integration

## Overview

This branch adds **GPU-direct EFA RDMA** support to DeepEP using
[efa-dp-direct](https://github.com/amzn/efa-dp-direct), Amazon's library
for posting EFA work requests directly from CUDA kernels.

### Architecture Comparison

```
Original (NVSHMEM IBGDA on InfiniBand):
  GPU kernel → MLX5 WQE → IB doorbell     (~118us LL latency)

NVSHMEM on EFA (current whn09 fork):
  GPU kernel → nvshmem_putmem → CPU proxy → ibverbs → EFA NIC   (~150-230us)

NEW: efa-dp-direct on EFA (this branch):
  GPU kernel → EFA WQE → EFA doorbell     (target: <120us)
```

The key insight: efa-dp-direct provides the same GPU-direct doorbell ringing
capability on EFA that NVSHMEM IBGDA provides on InfiniBand, eliminating the
CPU proxy bottleneck.

## What Changed

### New Files

| File | Description |
|------|-------------|
| `csrc/kernels/efa_dp_direct_device.cuh` | Device-side RDMA primitives using efa-dp-direct |
| `csrc/kernels/efa_dp_direct_runtime.cuh` | Host-side runtime API declaration |
| `csrc/kernels/efa_dp_direct_runtime.cu` | Host-side EFA init, QP creation, memory registration |

### Modified Files

| File | Changes |
|------|---------|
| `csrc/kernels/configs.cuh` | Added `USE_EFA_DP_DIRECT` compile-time switch |
| `csrc/kernels/internode_ll.cu` | `#ifdef` to include efa-dp-direct device header |
| `csrc/kernels/internode.cu` | `#ifdef` for EFA-DP path + placeholder types |
| `csrc/kernels/runtime.cu` | EFA-DP-direct init/alloc/free/barrier backend |
| `setup.py` | `USE_EFA_DP_DIRECT=1` build flag, efa-dp-direct linking |

### Primitive Mapping

| NVSHMEM IBGDA | efa-dp-direct | Notes |
|---------------|---------------|-------|
| `nvshmemi_ibgda_put_nbi_warp()` | `efa_cuda_init_rdma_write_wr()` + `efa_cuda_flush_sq_wrs()` | Direct RDMA Write WQE to EFA SQ |
| `nvshmemi_ibgda_amo_nonfetch_add()` | RDMA Write (inline 4 bytes) | EFA SRD doesn't support atomics |
| `nvshmemi_ibgda_quiet()` | `__threadfence_system()` (CQ polling TODO) | Needs dmabuf for GPU CQ access |
| `nvshmemi_ibgda_rma_p()` | RDMA Write (inline 4 bytes) | Single int write |
| `nvshmemx_barrier_all_block()` | Custom barrier via atomics | Already exists in DeepEP |
| `nvshmem_sync()` | `__threadfence_system()` | Placeholder |

## Build

### Prerequisites

- CUDA Toolkit 12.0+ (or 13.0 for B200)
- rdma-core 59+ (with `efadv_query_qp_wqs()` support)
- libibverbs-dev, libefa
- efa-dp-direct library (built from source)
- **root access** (needed for `cudaHostRegister(IoMemory)` on BAR regions)

### Build efa-dp-direct

```bash
git clone https://github.com/amzn/efa-dp-direct.git
cd efa-dp-direct/CUDA && make
```

### Build DeepEP with EFA-DP-direct

```bash
cd DeepEP
USE_EFA_DP_DIRECT=1 \
EFA_DP_DIRECT_DIR=/path/to/efa-dp-direct \
CUDA_HOME=/usr/local/cuda \
python setup.py install
```

Environment variables:
- `USE_EFA_DP_DIRECT=1` — Enable GPU-direct EFA path
- `EFA_DP_DIRECT_DIR` — Path to efa-dp-direct source (default: `third-party/efa-dp-direct`)
- `EFA_HOME` — Path to EFA installation (default: `/opt/amazon/efa`)

### Build without EFA-DP (NVSHMEM path)

```bash
# Original NVSHMEM path (unchanged)
NVSHMEM_DIR=/path/to/nvshmem python setup.py install
```

## Current Status

### Working
- [x] Device-side RDMA Write via efa-dp-direct (validated in E2E test)
- [x] Build system with `USE_EFA_DP_DIRECT` flag
- [x] Compile-time backend selection (NVSHMEM / EFA-DP / disabled)
- [x] Drop-in replacement for `nvshmemi_ibgda_put_nbi_warp()`
- [x] Signal replacement for `nvshmemi_ibgda_amo_nonfetch_add()` (RDMA Write)
- [x] Host-side EFA runtime (device open, QP creation, MR registration)

### TODO
- [ ] **CQ GPU access** via dmabuf — needed for `nvshmemi_ibgda_quiet()` from GPU
- [ ] **OOB exchange** of QP info via torch.distributed (currently manual)
- [ ] **Multi-QP batching** — submit multiple WRs in single doorbell ring
- [ ] **Performance benchmarking** — compare with NVSHMEM EFA and IBGDA paths
- [ ] **Production permissions** — `cudaHostRegister(IoMemory)` requires root
- [ ] **Normal mode** (internode.cu) — full `nvshmem_team_t` replacement

## Key Findings from Validation

1. **`efadv_query_qp_wqs()`** (rdma-core 59+) returns raw SQ/RQ buffer and doorbell pointers
2. **`cudaHostRegister(IoMemory)` + sudo** enables GPU access to BAR-mapped regions
3. **CQ buffer** cannot be registered via `cudaHostRegister` — needs dmabuf path
4. **EFA SRD QP** must use `efadv_create_qp_ex()` with `IBV_QP_EX_WITH_RDMA_WRITE`
5. **No nvidia-peermem** needed for GPU → EFA BAR writes
6. **GPU and CPU posting cannot be mixed** on the same QP

## Test Environment

- Instance: p6-b200.48xlarge × 2
- GPU: NVIDIA B200 183GB (SM 10.0, CUDA 13.0)
- Network: EFA 400Gb/s × 8/node
- EFA driver: 2.17.3g
- rdma-core: 59.amzn0-1

## References

- [efa-dp-direct](https://github.com/amzn/efa-dp-direct)
- [E2E validation tests](https://github.com/whn09/efa-dp-direct/tree/master/tests)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
