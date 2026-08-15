/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */
#ifndef ACC_OFFLOAD_SPARSE_KV_PLAN_RUNTIME_H
#define ACC_OFFLOAD_SPARSE_KV_PLAN_RUNTIME_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif // ACC_OFFLOAD_SPARSE_KV_PLAN_RUNTIME_H
