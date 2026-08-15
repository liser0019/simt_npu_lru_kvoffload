/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy at http://license.coscl.org.cn/MulanPSL2.
 */

#ifdef MF_ACC_OFFLOAD_PRODUCTION
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifdef MF_ACC_OFFLOAD_PRODUCTION
#include "acc_offload_lru_compact_fused_v3.h"
#include "acc_offload_lru_resident_addrs_mixed_parallel.h"
#include "acc_offload_sparse_kv_plan_runtime.h"
#include "acc_offload_sparse_kv_transfer_runtime.h"
#endif
#include "acc_offload_sparse_kv_runtime.h"
#include "acc_offload_operators.h"

#ifdef MF_ACC_OFFLOAD_PRODUCTION
namespace {

// Compact V3 is intentionally specialized for the production state shape.
// Keeping this check on the Host prevents accidental launches with a UB layout
// that was compiled for different dimensions.
bool IsCompactV3Shape(int64_t numReqs, int64_t topk,
                      int64_t capacity, int64_t maxToken)
{
    return numReqs > 0 && topk == 2048 && capacity == 4096 &&
           maxToken == 4096;
}

// ResidentAddrs Parallel V2 keeps the original one-row VF as its batch=1 fast
// path and uses a two-pass multi-row VF for dynamic num_reqs.  num_tokens is an
// int32 scalar, so also reject shapes whose maximum possible task count cannot
// be represented without overflow.
bool IsResidentAddrsParallelShape(int64_t numReqs, int64_t topk)
{
    constexpr int64_t maxTaskCount = 2147483647LL;
    return numReqs > 0 && topk > 0 && topk <= 2048 &&
           numReqs <= maxTaskCount / (2 * topk);
}

uint32_t GetLruV3ProfileEnabled()
{
    static const uint32_t enabled = []() {
        const char *value = std::getenv("MF_LRU_COMPACT_V3_PROFILE");
        return value != nullptr && std::strcmp(value, "1") == 0 ? 1U : 0U;
    }();
    return enabled;
}

void LogProductionBackendsOnce()
{
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }
    std::fprintf(stderr,
        "[ACC_OFFLOAD_PRODUCTION] "
        "compact=mixed_ub_fused_v3 "
        "resident_addrs=mixed_simt_parallel "
        "sparse_copy=aiv_datacopypad\n");
    std::fflush(stderr);
}

void LogSparseKvRuntimeOnce()
{
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }
    std::fprintf(stderr,
        "[SPARSE_KV_LOAD_RUNTIME] api=one plan=mixed_simt_hash "
        "transfer=aiv_datacopypad descriptors=none\n");
    std::fflush(stderr);
}

void LogCompactFallbackOnce(int64_t numReqs, int64_t topk,
                            int64_t capacity, int64_t maxToken)
{
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }
    std::fprintf(stderr,
        "[SIMT_LRU_PLAN_FALLBACK] Compact V3 requires "
        "topk=2048 capacity=4096 max_token=4096; using generic SIMT: "
        "num_reqs=%lld topk=%lld capacity=%lld max_token=%lld\n",
        static_cast<long long>(numReqs), static_cast<long long>(topk),
        static_cast<long long>(capacity), static_cast<long long>(maxToken));
    std::fflush(stderr);
}

void LogResidentAddrsFallbackOnce(int64_t numReqs, int64_t topk)
{
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }
    std::fprintf(stderr,
        "[RESIDENT_ADDRS_FALLBACK] mixed_simt_parallel requires "
        "num_reqs>0, 0<topk<=2048 and int32-safe task count; "
        "using generic SIMT: "
        "num_reqs=%lld topk=%lld\n",
        static_cast<long long>(numReqs), static_cast<long long>(topk));
    std::fflush(stderr);
}

} // namespace
#endif

extern "C" {

void AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs,
                          uint32_t *lenPtrs, uint32_t *sizePtr,
                          uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);
    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream]() -> int {
        OffloadOpsSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("acc_offload_sparse_copy", callback);
}

void AccOffloadLruCompact(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
    uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
    uint64_t miss_slots, uint64_t token_mark_workspace,
    uint64_t token_pos_workspace, uint64_t epochs, int64_t num_reqs,
    int64_t topk, int64_t capacity, int64_t max_token, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);
    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [=]() -> int {
#ifdef MF_ACC_OFFLOAD_PRODUCTION
        LogProductionBackendsOnce();
        if (IsCompactV3Shape(num_reqs, topk, capacity, max_token)) {
            for (int64_t row = 0; row < num_reqs; ++row) {
                OffloadOpsLruCompactFusedV3(
                    req_ids, last_req_ids, topk_indices, stable_prefix_lens,
                    slot_to_token, lru_slots, current_slots, miss_count,
                    miss_tokens, miss_slots, token_mark_workspace,
                    token_pos_workspace, epochs, row, topk, capacity,
                    max_token, npuStream, GetLruV3ProfileEnabled());
            }
            return 0;
        }

        LogCompactFallbackOnce(num_reqs, topk, capacity, max_token);
        OffloadOpsLruCompact(
            req_ids, last_req_ids, topk_indices, stable_prefix_lens,
            slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
            miss_slots, token_mark_workspace, token_pos_workspace, epochs,
            num_reqs, topk, capacity, max_token, npuStream);
#else
        OffloadOpsLruCompact(
            req_ids, last_req_ids, topk_indices, stable_prefix_lens,
            slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
            miss_slots, token_mark_workspace, token_pos_workspace, epochs,
            num_reqs, topk, capacity, max_token, npuStream);
#endif
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "acc_offload_lru_compact", callback);
}

void AccOffloadComputeLruResidentAddrs(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
    uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v,
    int64_t gvas_k_base, int64_t gvas_v_base, int64_t addr_k_base,
    int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs,
    int64_t topk, int64_t max_num_blocks, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);
    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [=]() -> int {
#ifdef MF_ACC_OFFLOAD_PRODUCTION
        LogProductionBackendsOnce();
        if (IsResidentAddrsParallelShape(num_reqs, topk)) {
            OffloadOpsComputeLruResidentAddrsMixedParallel(
                miss_count, miss_tokens, miss_slots, block_table,
                gvas_buffer, addr_buffer, size_buffer, num_tokens_buffer,
                block_size, token_size_bytes_k, token_size_bytes_v,
                gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
                resident_capacity, num_reqs, topk, max_num_blocks,
                npuStream);
            return 0;
        }

        LogResidentAddrsFallbackOnce(num_reqs, topk);
        OffloadOpsComputeLruResidentAddrs(
            miss_count, miss_tokens, miss_slots, block_table, gvas_buffer,
            addr_buffer, size_buffer, num_tokens_buffer, block_size,
            token_size_bytes_k, token_size_bytes_v, gvas_k_base,
            gvas_v_base, addr_k_base, addr_v_base, resident_capacity,
            num_reqs, topk, max_num_blocks, npuStream);
#else
        OffloadOpsComputeLruResidentAddrs(
            miss_count, miss_tokens, miss_slots, block_table, gvas_buffer,
            addr_buffer, size_buffer, num_tokens_buffer, block_size,
            token_size_bytes_k, token_size_bytes_v, gvas_k_base,
            gvas_v_base, addr_k_base, addr_v_base, resident_capacity,
            num_reqs, topk, max_num_blocks, npuStream);
#endif
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "acc_offload_compute_lru_resident_addrs", callback);
}

#ifdef MF_ACC_OFFLOAD_PRODUCTION
void AccOffloadSparseKvLoadRuntime(
    const sparse_kv_load_runtime_params_t *params, uint8_t devIdx)
{
    if (params == nullptr) {
        return;
    }
    sparse_kv_load_runtime_params_t value = *params;
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);
    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    // RunOpApiV2 executes the callback while this stack frame is alive, but
    // capture a POD copy rather than the caller's pointer to make ownership
    // explicit.  The two launches share npuStream and intentionally contain no
    // synchronize: stream ordering is the Plan -> Transfer dependency.
    auto callback = [value, npuStream]() -> int {
        LogSparseKvRuntimeOnce();
        OffloadOpsSparseKvPlanRuntime(
            value.req_ids, value.last_req_ids, value.topk_indices,
            value.stable_prefix_lens, value.slot_to_token, value.lru_slots,
            value.current_slots, value.miss_count, value.miss_tokens,
            value.miss_slots, value.compact_workspace,
            value.compact_workspace_bytes, value.num_reqs, value.topk,
            value.capacity, value.max_token, npuStream);
        OffloadOpsSparseKvTransferRuntime(
            value.miss_count, value.miss_tokens, value.miss_slots,
            value.block_table, value.host_k_base, value.host_v_base,
            value.device_k_base, value.device_v_base, value.num_reqs,
            value.topk, value.capacity, value.max_num_blocks,
            value.block_size, value.token_size_bytes_k,
            value.token_size_bytes_v, npuStream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "acc_offload_sparse_kv_load_runtime", callback);
}

void AccOffloadSparseKvPlanFsaRuntime(
    const sparse_kv_plan_fsa_runtime_params_t *params, uint8_t devIdx)
{
    if (params == nullptr) {
        return;
    }
    sparse_kv_plan_fsa_runtime_params_t value = *params;
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);
    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    // Mapping and FSA planning are enqueued on the same current stream.  This
    // API is intentionally plan-only: it never invokes TransferRuntime and it
    // never synchronizes or exposes a Host plan bridge.
    auto callback = [value, npuStream]() -> int {
        OffloadOpsSparseKvPlanFsaRuntime(
            value.req_ids, value.last_req_ids, value.topk_indices,
            value.stable_prefix_lens, value.visible_seq_lens,
            value.slot_to_token, value.lru_slots, value.current_slots,
            value.miss_count, value.miss_tokens, value.miss_slots,
            value.compact_workspace, value.compact_workspace_bytes,
            value.row_map_workspace, value.row_map_workspace_bytes,
            value.encoded_plan, value.current_linear_slots,
            value.num_logical_rows, value.physical_row_capacity, value.topk,
            value.capacity, value.max_token, value.encoded_plan_stride,
            npuStream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "acc_offload_sparse_kv_plan_fsa_runtime", callback);
}
#endif // MF_ACC_OFFLOAD_PRODUCTION

} // extern "C"
