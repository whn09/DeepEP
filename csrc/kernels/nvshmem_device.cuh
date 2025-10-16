#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace deep_ep {

// 替代ibgda_get_p2p_ptr函数
// 获取远程PE的P2P（点对点）指针
// 如果返回0，表示没有P2P连接，需要使用RDMA操作
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // 本地rank，不需要映射
    if (rank == dst_rank)
        return ptr;

    // 使用nvshmem_ptr获取远程指针
    // 注意：在EFA环境中，这可能返回NULL，因为EFA可能不支持直接内存访问
    void* remote_ptr = nvshmem_ptr(reinterpret_cast<void*>(ptr), dst_rank);
    if (remote_ptr == NULL)
        return 0;

    return reinterpret_cast<uint64_t>(remote_ptr);
}

// 替代nvshmemi_ibgda_put_nbi_warp函数
template <bool kAlwaysDoPostSend = false>
__device__ __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    // 在EFA环境中，我们应该使用标准的nvshmem_putmem_nbi函数
    // 对于warp级别的操作，使用nvshmemx_uint64_put_nbi_warp
    if (lane_id == 0) {  // 只让一个线程执行put操作
        nvshmem_putmem_nbi(reinterpret_cast<void*>(req_rptr),
                          reinterpret_cast<const void*>(req_lptr),
                          bytes,
                          dst_pe);
    }
    // 确保warp中的所有线程同步
    __syncwarp();
}

// 替代nvshmemi_ibgda_amo_nonfetch_add函数
// 原子非获取加法操作：将value加到远程PE的rptr指向的内存位置
// 这是一个单边操作，不返回原值
__device__ __forceinline__ void
nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned int*>(rptr), value);
    } else {
        // 使用nvshmem_int_atomic_add替代
        nvshmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
    }
}

// 实现 nvshmemi_ibgda_rma_p 函数
// RMA Put操作：将单个int值写入远程PE的指定地址
// 这是一个阻塞的单边写操作，类似于RDMA的内联写入
//
// 参数：
//   - rptr: 远程PE上的目标地址（指向int类型）
//   - value: 要写入的int值
//   - dst_pe: 目标PE（进程）的编号
//   - qp_id: 队列对ID（在nvshmem中不使用，保留参数以保持接口兼容）
//   - imm: 立即数（可选，用于带立即数的写操作，在nvshmem标准API中不使用）
//
// 功能说明：
// 此函数是InfiniBand IBGDA（GPU Direct Async）RMA Put操作的替代实现。
// 原IBGDA版本直接操作InfiniBand QP（Queue Pair）硬件，构造RDMA写入WQE（Work Queue Element）。
// 本实现使用NVSHMEM的标准API来实现相同的语义：单边写入一个int值到远程PE。
//
// 在EFA环境中的工作方式：
// - NVSHMEM会自动选择最优传输路径（EFA RDMA或其他底层传输）
// - 对于单个int的写入，nvshmem_int_p是最高效的选择
// - 操作完成后，远程PE可以看到更新后的值
__device__ __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    // 使用nvshmem的标准put操作来实现单个int值的写入
    // nvshmem_int_p 是一个阻塞的RMA put操作，用于写入单个int值
    // 这个操作在NVSHMEM内部会被转换为适合底层网络的传输操作（例如EFA的RDMA写入）
    nvshmem_int_p(rptr, value, dst_pe);

    // 注意：原IBGDA实现中的imm参数用于RDMA写入时携带立即数
    // 在NVSHMEM标准API中，没有直接支持立即数的接口
    // 如果需要立即数功能，可能需要通过其他机制（如额外的信号传递）来实现
}

} // namespace deep_ep
