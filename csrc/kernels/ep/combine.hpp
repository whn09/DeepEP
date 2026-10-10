#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_ep/impls/ep/proxy_ring.cuh>

#include "../../runtime/jit.hpp"
#include "kernel_select.hpp"

namespace deep_ep::ep {

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            int* token_map_at_dispatch,
                            const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_combined_tokens,
                            const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    // Decide warps
    const bool use_ordered_kernel = use_ordered_hybrid_kernel();
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        if (use_ordered_kernel) {
            // The ordered kernel has no proxy warp and only carves TMA buffers out of shmem
            num_warps = num_scaleup_warps + num_forward_warps;
            EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                           "Invalid combine SM count, please try to match your dispatch config");
        } else {
            // Scale-up + forward warps, plus one proxy warp
            const auto num_data_warps = num_scaleup_warps + num_forward_warps;
            num_warps = num_data_warps + 1;
            EP_HOST_ASSERT(num_warps * 32 <= 1024 and
                           "combine warp count (scale-up + forward + proxy) exceeds the "
                           "1024-thread block limit; use at least num_channels / 15 SMs");

            // TMA buffers and the proxy hand-off rings (placed after the TMA region) share the
            // dynamic shared memory; the dispatch-side channel tuner sized against the same total
            const int64_t tma_smem_bytes = static_cast<int64_t>(num_data_warps) * token_layout.get_num_bytes<true>();
            const int64_t proxy_ring_bytes = ProxyRingLayout::get_num_bytes(num_forward_warps, kProxyRingDepthDefault);
            EP_HOST_ASSERT(tma_smem_bytes + proxy_ring_bytes <= num_smem_bytes and
                           "Combine TMA buffers + proxy rings exceed per-block shared memory");
        }
    }

    const auto num_threads = num_warps * 32;

    // Compile
    std::string header_name, func_name;
    if (num_scaleout_ranks == 1) {
        header_name = "combine";
        func_name = std::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                is_scaleup_nvlink,
                                use_expanded_layout, allow_multiple_reduction,
                                num_sms,
                                num_threads / 32,
                                num_scaleup_ranks * num_scaleout_ranks,
                                hidden,
                                num_max_tokens_per_rank,
                                num_experts,
                                num_topk,
                                num_qps, num_timeout_cycles);
    } else {
        header_name = use_ordered_kernel ? "hybrid_combine" : "hybrid_combine_unordered";
        func_name = std::format("{}<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                use_ordered_kernel ? "hybrid_combine_impl" : "hybrid_unordered_combine_impl",
                                use_expanded_layout, allow_multiple_reduction,
                                num_sms,
                                num_scaleup_warps, num_forward_warps,
                                num_scaleout_ranks, num_scaleup_ranks,
                                hidden,
                                num_max_tokens_per_rank,
                                num_experts,
                                num_topk,
                                num_qps,
                                num_timeout_cycles);
    }
    const auto kernel = jit->compile("combine", std::format(R"(
#include <deep_ep/impls/ep/{}.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::{});
}}
)", header_name, func_name));

    // Launch
    const auto options = deep_jit::cuda::LaunchOptions {
        .stream = stream.stream(),
        .num_smem_bytes = num_smem_bytes,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
        .cooperative = true,
    };
    if (num_scaleout_ranks == 1) {
        jit->launch(
            kernel, options,
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights),
            src_metadata, psum_num_recv_tokens_per_scaleup_rank,
            nccl_dev_comm, nccl_window,
            buffer, workspace,
            scaleup_rank_idx,
            num_reduced_tokens
        );
    } else if (use_ordered_kernel) {
        jit->launch(
            kernel, options,
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights),
            src_metadata,
            psum_num_recv_tokens_per_scaleup_rank,
            token_metadata_at_forward,
            channel_linked_list,
            nccl_dev_comm, nccl_window,
            buffer, workspace,
            scaleout_rank_idx, scaleup_rank_idx,
            num_reduced_tokens
        );
    } else {
        jit->launch(
            kernel, options,
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights),
            src_metadata,
            psum_num_recv_tokens_per_scaleup_rank,
            token_metadata_at_forward,
            channel_linked_list,
            static_cast<const int*>(token_map_at_dispatch),
            nccl_dev_comm, nccl_window,
            buffer, workspace,
            scaleout_rank_idx, scaleup_rank_idx,
            num_reduced_tokens, num_combined_tokens
        );
    }

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           const int& num_channels,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           int* token_map_at_dispatch,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Compile
    const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/combine_reduce_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", use_expanded_layout, allow_multiple_reduction,
        num_sms, num_threads / 32,
        num_scaleout_ranks, num_scaleup_ranks,
        hidden,
        num_max_tokens_per_rank,
        num_experts, num_topk,
        num_channels,
        use_ordered_hybrid_kernel()));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .enable_pdl = true,
        },
        static_cast<nv_bfloat16*>(combined_x),
        combined_topk_weights,
        combined_topk_idx,
        reduce_buffer,
        bias_0, bias_1,
        static_cast<const int*>(token_map_at_dispatch),
        num_combined_tokens,
        scaleout_rank_idx, scaleup_rank_idx
    );
}

}  // namespace deep_ep::ep
