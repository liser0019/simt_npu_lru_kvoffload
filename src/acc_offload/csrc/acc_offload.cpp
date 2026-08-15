/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "acc_offload.h"
#include "acc_offload_entry_manager.h"
#include "acc_offload_define.h"

#include <climits>

using namespace ock::offload;

OFFLOAD_API int32_t offload_init(const offload_config_t &config)
{
    return AccOffloadEntryManager::Instance().Initialize(config);
}

OFFLOAD_API void offload_uninit()
{
    AccOffloadEntryManager::Instance().UnInitalize();
}

OFFLOAD_API uint64_t offload_malloc(uint64_t size, uint64_t flags)
{
    (void)flags;
    auto ptr = AccOffloadEntryManager::Instance().MallocHost(size);
    if (ptr == nullptr) {
        OFFLOAD_LOG_ERROR("offload_malloc failed, size:" << size);
        return 0;
    }

    return reinterpret_cast<uint64_t>(ptr);
}

OFFLOAD_API void offload_free(uint64_t ptr, uint64_t flags)
{
    (void)flags;
    AccOffloadEntryManager::Instance().FreeHost(reinterpret_cast<void *>(ptr));
}

OFFLOAD_API uint64_t offload_get_device_address(uint64_t ptr, uint64_t size)
{
    return AccOffloadEntryManager::Instance().GetDeviceAddress(reinterpret_cast<void *>(ptr), size);
}

OFFLOAD_API int32_t offload_sparse_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t sizePtr,
                                        uint16_t deviceId)
{
    auto srcPtrs = reinterpret_cast<uint64_t *>(srcPtr);
    auto dstPtrs = reinterpret_cast<uint64_t *>(dstPtr);
    auto lenPtrs = reinterpret_cast<uint32_t *>(lenPtr);
    auto sizePtr_ = reinterpret_cast<uint32_t *>(sizePtr);

    return AccOffloadEntryManager::Instance().SparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr_,
                                                         static_cast<uint8_t>(deviceId));
}

OFFLOAD_API int32_t offload_lru_resident_compact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                                  uint64_t stable_prefix_lens, uint64_t slot_to_token,
                                                  uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
                                                  uint64_t miss_tokens, uint64_t miss_slots, uint64_t token_mark_workspace,
                                                  uint64_t token_pos_workspace, uint64_t epochs, int64_t num_reqs,
                                                  int64_t topk, int64_t capacity, int64_t max_token,
                                                  uint16_t deviceId)
{
    return AccOffloadEntryManager::Instance().LruResidentCompact(
        req_ids, last_req_ids, topk_indices, stable_prefix_lens, slot_to_token, lru_slots, current_slots, miss_count,
        miss_tokens, miss_slots, token_mark_workspace, token_pos_workspace, epochs, num_reqs, topk, capacity, max_token,
        static_cast<uint8_t>(deviceId));
}

OFFLOAD_API int32_t offload_compute_lru_resident_addrs(uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                                        uint64_t block_table, uint64_t gvas_buffer,
                                                        uint64_t addr_buffer, uint64_t size_buffer,
                                                        uint64_t num_tokens_buffer, int32_t block_size,
                                                        int32_t token_size_bytes_k, int32_t token_size_bytes_v,
                                                        int64_t gvas_k_base, int64_t gvas_v_base,
                                                        int64_t addr_k_base, int64_t addr_v_base,
                                                        int32_t resident_capacity, int64_t num_reqs, int64_t topk,
                                                        int64_t max_num_blocks, uint16_t deviceId)
{
    return AccOffloadEntryManager::Instance().ComputeLruResidentAddrs(
        miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer, size_buffer, num_tokens_buffer,
        block_size, token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks, static_cast<uint8_t>(deviceId));
}

OFFLOAD_API uint64_t offload_get_sparse_kv_plan_workspace_size(
    int64_t num_reqs, int64_t topk, int64_t capacity)
{
    return sparse_kv_plan_workspace_size_bytes(num_reqs, topk, capacity);
}

OFFLOAD_API uint64_t offload_get_sparse_kv_fsa_row_map_workspace_size(
    int64_t num_logical_rows, int64_t physical_row_capacity)
{
    return sparse_kv_fsa_row_map_workspace_size_bytes(
        num_logical_rows, physical_row_capacity);
}

OFFLOAD_API uint64_t offload_get_sparse_kv_fsa_plan_row_stride(int64_t topk)
{
    return sparse_kv_fsa_plan_row_stride_int16(topk);
}

namespace {
bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > UINT64_MAX / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool AddressRangeFits(uint64_t base, uint64_t bytes)
{
    return bytes > 0 && base <= UINT64_MAX - (bytes - 1U);
}

bool ValidateSparseKvRuntimeParams(
    const sparse_kv_load_runtime_params_t &params)
{
    const uint64_t requiredWorkspace =
        sparse_kv_plan_workspace_size_bytes(
            params.num_reqs, params.topk, params.capacity);
    const bool pointersValid =
        params.req_ids != 0 && params.last_req_ids != 0 &&
        params.topk_indices != 0 && params.stable_prefix_lens != 0 &&
        params.slot_to_token != 0 && params.lru_slots != 0 &&
        params.current_slots != 0 && params.miss_count != 0 &&
        params.miss_tokens != 0 && params.miss_slots != 0 &&
        params.compact_workspace != 0 && params.block_table != 0 &&
        params.host_k_base != 0 && params.host_v_base != 0 &&
        params.device_k_base != 0 && params.device_v_base != 0;
    const bool dimensionsValid =
        params.num_reqs > 0 && params.topk > 0 &&
        params.topk <= INT32_MAX / 2 && params.capacity > 0 &&
        params.capacity <= INT32_MAX && params.max_token > 0 &&
        params.max_token <= INT32_MAX && params.max_num_blocks > 0 &&
        params.block_size > 0 && params.token_size_bytes_k > 0 &&
        params.token_size_bytes_v > 0;
    if (!pointersValid || !dimensionsValid || requiredWorkspace == 0 ||
        params.compact_workspace_bytes < requiredWorkspace) {
        return false;
    }
    if (!sparse_kv_runtime_block_table_covers_tokens(
            params.max_token, params.block_size,
            params.max_num_blocks)) {
        return false;
    }
    if (params.num_reqs > INT64_MAX / params.topk ||
        params.num_reqs > INT64_MAX / params.capacity ||
        params.num_reqs > INT64_MAX / params.max_num_blocks) {
        return false;
    }

    uint64_t residentTokens = 0;
    uint64_t residentKBytes = 0;
    uint64_t residentVBytes = 0;
    if (!CheckedMultiply(static_cast<uint64_t>(params.num_reqs),
                         static_cast<uint64_t>(params.capacity),
                         residentTokens) ||
        !CheckedMultiply(residentTokens,
                         static_cast<uint64_t>(params.token_size_bytes_k),
                         residentKBytes) ||
        !CheckedMultiply(residentTokens,
                         static_cast<uint64_t>(params.token_size_bytes_v),
                         residentVBytes) ||
        !AddressRangeFits(params.device_k_base, residentKBytes) ||
        !AddressRangeFits(params.device_v_base, residentVBytes)) {
        return false;
    }

    // block_table entries are int32.  Validate the largest address the
    // Transfer kernel could form even when a malformed table contains
    // INT32_MAX, so unsigned device-side pointer arithmetic cannot wrap.
    uint64_t blockBytesK = 0;
    uint64_t blockBytesV = 0;
    uint64_t sourceKBytes = 0;
    uint64_t sourceVBytes = 0;
    const uint64_t maximumPhysicalBlocks =
        static_cast<uint64_t>(INT32_MAX) + 1U;
    if (!CheckedMultiply(static_cast<uint64_t>(params.block_size),
                         static_cast<uint64_t>(params.token_size_bytes_k),
                         blockBytesK) ||
        !CheckedMultiply(static_cast<uint64_t>(params.block_size),
                         static_cast<uint64_t>(params.token_size_bytes_v),
                         blockBytesV) ||
        !CheckedMultiply(maximumPhysicalBlocks, blockBytesK,
                         sourceKBytes) ||
        !CheckedMultiply(maximumPhysicalBlocks, blockBytesV,
                         sourceVBytes)) {
        return false;
    }
    return AddressRangeFits(params.host_k_base, sourceKBytes) &&
           AddressRangeFits(params.host_v_base, sourceVBytes);
}

bool ValidateSparseKvFsaRuntimeParams(
    const sparse_kv_plan_fsa_runtime_params_t &params)
{
    const uint64_t requiredCompact =
        sparse_kv_plan_workspace_size_bytes(
            params.num_logical_rows, params.topk, params.capacity);
    const uint64_t requiredRowMap =
        sparse_kv_fsa_row_map_workspace_size_bytes(
            params.num_logical_rows, params.physical_row_capacity);
    const uint64_t requiredStride =
        sparse_kv_fsa_plan_row_stride_int16(params.topk);
    const bool pointersValid =
        params.req_ids != 0 && params.last_req_ids != 0 &&
        params.topk_indices != 0 && params.stable_prefix_lens != 0 &&
        params.visible_seq_lens != 0 && params.slot_to_token != 0 &&
        params.lru_slots != 0 && params.current_slots != 0 &&
        params.miss_count != 0 && params.miss_tokens != 0 &&
        params.miss_slots != 0 && params.compact_workspace != 0 &&
        params.row_map_workspace != 0 && params.encoded_plan != 0 &&
        params.current_linear_slots != 0;
    const bool dimensionsValid =
        params.num_logical_rows > 0 &&
        params.physical_row_capacity >= params.num_logical_rows &&
        params.physical_row_capacity <= INT16_MAX &&
        params.topk > 0 && params.topk <= 2048 &&
        params.capacity > 1 && params.capacity <= INT16_MAX &&
        params.max_token > 0 && params.max_token <= INT32_MAX &&
        params.encoded_plan_stride >= static_cast<int64_t>(requiredStride);
    if (!pointersValid || !dimensionsValid || requiredCompact == 0 ||
        requiredRowMap == 0 || requiredStride == 0 ||
        params.compact_workspace_bytes < requiredCompact ||
        params.row_map_workspace_bytes < requiredRowMap) {
        return false;
    }
    if (params.num_logical_rows > INT64_MAX / params.topk ||
        params.physical_row_capacity > INT64_MAX / params.capacity ||
        params.num_logical_rows > INT64_MAX / params.encoded_plan_stride) {
        return false;
    }
    return true;
}
} // namespace

OFFLOAD_API int32_t offload_sparse_kv_load_runtime(
    const sparse_kv_load_runtime_params_t *params, uint16_t deviceId)
{
    if (params == nullptr || deviceId > UINT8_MAX ||
        !ValidateSparseKvRuntimeParams(*params)) {
        OFFLOAD_LOG_ERROR("invalid sparse_kv_load_runtime parameters");
        return OFFLOAD_ERROR;
    }
    return AccOffloadEntryManager::Instance().SparseKvLoadRuntime(
        *params, static_cast<uint8_t>(deviceId));
}


OFFLOAD_API int32_t offload_sparse_kv_plan_fsa_runtime(
    const sparse_kv_plan_fsa_runtime_params_t *params, uint16_t deviceId)
{
    if (params == nullptr || deviceId > UINT8_MAX ||
        !ValidateSparseKvFsaRuntimeParams(*params)) {
        OFFLOAD_LOG_ERROR("invalid sparse_kv_plan_fsa_runtime parameters");
        return OFFLOAD_ERROR;
    }
    return AccOffloadEntryManager::Instance().SparseKvPlanFsaRuntime(
        *params, static_cast<uint8_t>(deviceId));
}
