# DeepEP EFA Migration Guide

## 概述

本指南说明如何将 DeepEP 从 InfiniBand IBGDA 迁移到 Amazon EFA (Elastic Fabric Adapter)。

## 当前状态分析

### 文件结构
```
csrc/kernels/
├── internode.cu          # 跨节点通信主要实现 (使用 nvshmem_device.cuh)
├── internode_ll.cu       # 低延迟模式实现 (使用 nvshmem_device.cuh)
├── ibgda_device.cuh      # 原始 InfiniBand GDMA 实现 (参考)
├── nvshmem_device.cuh    # 当前兼容层 (简化的 NVSHMEM 封装)
├── efa_device.cuh        # 原始 EFA 实现 (有问题)
└── efa_device_fixed.cuh  # 修复后的 EFA 实现
```

### 问题诊断

#### efa_device.cuh 中的主要问题:

1. **架构问题**:
   - 试图模拟 IBGDA 的底层硬件接口
   - 定义了不必要的结构体 (efa_request_t, nvshmemi_efa_device_state_t等)
   - 尝试访问未定义的 NVSHMEM 内部状态 (nvshmemi_device_state_d)

2. **实现问题**:
   - `efa_get_symmetric_heap_ptr()` 使用了不存在的内部结构
   - 静态设备数组可能导致编译错误
   - 过度复杂的 warp 分片逻辑

3. **概念问题**:
   - EFA 不需要手动管理 QP (Queue Pairs)
   - NVSHMEM 已经抽象了底层传输细节

## 推荐的迁移方案

### 方案 1: 使用现有的 nvshmem_device.cuh (推荐)

**优点**:
- 最简单,已经在使用
- 代码已经测试过
- NVSHMEM 会自动选择最佳传输层 (包括 EFA)

**实施步骤**:
1. 保持 internode.cu 和 internode_ll.cu 使用 nvshmem_device.cuh
2. 确保 NVSHMEM 编译时启用 EFA 支持
3. 无需修改代码

**配置**:
```bash
# 编译 NVSHMEM 时
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_LIBFABRIC_SUPPORT=1
export NVSHMEM_DISABLE_CUDA_VMM=0

# 运行时
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_DISABLE_IB=1
export FI_PROVIDER=efa
```

### 方案 2: 使用修复后的 efa_device_fixed.cuh

**优点**:
- 明确针对 EFA 优化
- 可以添加 EFA 特定的性能调优
- 更清晰的代码结构

**实施步骤**:

1. 替换头文件引用:
```cpp
// 在 internode.cu 中
// 旧的
#include "nvshmem_device.cuh"

// 新的
#include "efa_device_fixed.cuh"
```

2. 同样在 internode_ll.cu 中进行替换

3. 验证编译和运行

## 关键 API 映射

### IBGDA → EFA/NVSHMEM

| IBGDA 函数 | EFA/NVSHMEM 等价函数 | 说明 |
|-----------|-------------------|------|
| `nvshmemi_ibgda_put_nbi_warp` | `nvshmem_putmem_nbi` | 非阻塞 PUT 操作 |
| `nvshmemi_ibgda_amo_nonfetch_add` | `nvshmem_int_atomic_add` | 原子加法操作 |
| `nvshmemi_ibgda_rma_p` | `nvshmem_int_p` | 单个 int 的 PUT 操作 |
| `nvshmemi_ibgda_quiet` | `nvshmem_quiet` | 等待所有操作完成 |
| `nvshmemi_get_p2p_ptr` | `nvshmem_ptr` | 获取 P2P 指针 |

## 性能优化建议

### 1. P2P vs RDMA 路径

efa_device_fixed.cuh 实现了智能路径选择:

```cpp
// 首先尝试 NVLink P2P (同节点 GPU)
uint64_t p2p_ptr = nvshmemi_get_p2p_ptr(req_rptr, nvshmem_my_pe(), dst_pe);
if (p2p_ptr != 0) {
    // 使用快速的直接内存访问
    memcpy(...);
} else {
    // 使用 EFA RDMA
    nvshmem_putmem_nbi(...);
}
```

### 2. 批处理操作

对于大量小消息,考虑批处理:

```cpp
// 发送多个消息
for (int i = 0; i < n; i++) {
    nvshmem_putmem_nbi(...);
}
// 一次性等待所有完成
nvshmem_quiet();
```

### 3. Warp 级优化

在 warp 内只让一个线程执行网络操作,避免重复:

```cpp
if (lane_id == 0) {
    nvshmem_putmem_nbi(...);
}
__syncwarp();
```

## 测试和验证

### 编译测试

```bash
cd /home/ubuntu/DeepEP
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
make -j
```

### 运行测试

```bash
# 单节点测试 (使用 NVLink P2P)
mpirun -np 8 ./test_program

# 多节点测试 (使用 EFA)
mpirun -np 16 --hostfile hosts ./test_program
```

### 验证 EFA 使用

```bash
# 检查是否使用了 EFA
export NVSHMEM_DEBUG=1
export FI_LOG_LEVEL=info

# 应该看到类似输出:
# [INFO] libfabric provider: efa
# [INFO] Using EFA device: rdmap0s8
```

## 故障排查

### 问题 1: 编译错误 "nvshmemi_device_state_d undefined"

**原因**: 尝试访问 NVSHMEM 内部结构
**解决**: 使用 efa_device_fixed.cuh,它只使用公共 API

### 问题 2: 运行时性能差

**检查**:
```bash
# 确认 EFA 被使用
fi_info -p efa

# 检查 GPU Direct RDMA 支持
nvidia-smi nvlink -s
```

### 问题 3: 通信超时

**可能原因**:
- EFA 网络配置问题
- 安全组设置阻止了通信
- MTU 大小不匹配

**解决**:
```bash
# 检查 EFA 设备
ibv_devinfo

# 测试基本连接
fi_pingpong -p efa
```

## 代码差异对比

### nvshmem_device.cuh vs efa_device_fixed.cuh

两者功能相同,主要区别:

1. **nvshmem_device.cuh**:
   - 更简洁
   - 适用于所有 NVSHMEM 支持的传输层
   - 已在生产中验证

2. **efa_device_fixed.cuh**:
   - 更详细的注释
   - 明确的 P2P 路径优化
   - 更容易理解和修改

## 推荐的实施路径

### 短期 (立即可用)
1. **不修改代码** - 继续使用 nvshmem_device.cuh
2. 正确配置 NVSHMEM 环境变量启用 EFA
3. 进行性能测试

### 中期 (可选优化)
1. 切换到 efa_device_fixed.cuh
2. 添加 EFA 特定的性能监控
3. 根据 profiling 结果优化

### 长期 (高级优化)
1. 根据工作负载特征定制传输策略
2. 实现自适应的批处理大小
3. 优化 GPU 内存对齐和数据布局

## 总结

**最简单的方案**: 保持当前代码不变,通过环境变量启用 EFA

**最佳实践方案**: 使用 efa_device_fixed.cuh,它提供了清晰的实现和优化机会

**关键点**:
- EFA 通过 NVSHMEM 的标准 API 使用,不需要特殊的底层接口
- P2P (NVLink) 路径应该优先于 RDMA 路径
- 正确的环境配置比代码修改更重要

## 参考资源

- [NVSHMEM Programming Guide](https://docs.nvidia.com/nvshmem/)
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Libfabric EFA Provider](https://github.com/ofiwg/libfabric/tree/main/prov/efa)
