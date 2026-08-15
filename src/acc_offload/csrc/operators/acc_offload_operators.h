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

#ifndef ACC_OFFLOAD_OPERATORS_H
#define ACC_OFFLOAD_OPERATORS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void OffloadOpsSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, void *stream);

void OffloadOpsLruCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                          uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
                          uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                          uint64_t token_mark_workspace, uint64_t token_pos_workspace, uint64_t epochs,
                          int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token, void *stream);

void OffloadOpsComputeLruResidentAddrs(uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                       uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
                                       uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
                                       int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base,
                                       int64_t gvas_v_base, int64_t addr_k_base, int64_t addr_v_base,
                                       int32_t resident_capacity, int64_t num_reqs, int64_t topk,
                                       int64_t max_num_blocks, void *stream);

void OffloadOpsSparseKvPlanRuntime(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t slot_to_token,
    uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
    uint64_t miss_tokens, uint64_t miss_slots, uint64_t compact_workspace,
    uint64_t compact_workspace_bytes, int64_t num_reqs, int64_t topk,
    int64_t capacity, int64_t max_token, void *stream);

void OffloadOpsSparseKvPlanFsaRuntime(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t visible_seq_lens,
    uint64_t slot_to_token, uint64_t lru_slots, uint64_t current_slots,
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t compact_workspace, uint64_t compact_workspace_bytes,
    uint64_t row_map_workspace, uint64_t row_map_workspace_bytes,
    uint64_t encoded_plan, uint64_t current_linear_slots,
    int64_t num_logical_rows, int64_t physical_row_capacity,
    int64_t topk, int64_t capacity, int64_t max_token,
    int64_t encoded_plan_stride, void *stream);

void OffloadOpsSparseKvTransferRuntime(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t block_table, uint64_t host_k_base, uint64_t host_v_base,
    uint64_t device_k_base, uint64_t device_v_base, int64_t num_reqs,
    int64_t topk, int64_t capacity, int64_t max_num_blocks,
    int32_t block_size, int32_t token_size_bytes_k,
    int32_t token_size_bytes_v, void *stream);

#ifdef __cplusplus
}
#endif

#endif // ACC_OFFLOAD_OPERATORS_H
