/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_SPARSE_KV_RUNTIME_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_SPARSE_KV_RUNTIME_H

#include <limits.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One logical SparseKvLoad submission.  All pointer fields are device-visible
 * addresses.  host_k_base/host_v_base are not ordinary CPU virtual addresses:
 * callers must obtain the registered-host DVA through get_device_address().
 *
 * The device executes two same-stream kernels:
 *   Plan     : TopK + resident/LRU state -> compact per-request miss plan.
 *   Transfer : miss plan -> address calculation -> AIV DataCopyPad H2D copy.
 *
 * No ResidentAddrs descriptor arrays are part of this ABI.
 */
typedef struct sparse_kv_load_runtime_params {
    uint64_t req_ids;
    uint64_t last_req_ids;
    uint64_t topk_indices;
    uint64_t stable_prefix_lens;
    uint64_t slot_to_token;
    uint64_t lru_slots;
    uint64_t current_slots;
    uint64_t miss_count;
    uint64_t miss_tokens;
    uint64_t miss_slots;
    uint64_t compact_workspace;
    uint64_t compact_workspace_bytes;
    /*
     * Row-major logical-to-physical block mapping.  For every valid miss,
     * token/block_size must address this row and the selected entry must be
     * non-negative.  A negative entry is invalid upstream runtime state; the
     * device Transfer kernel skips it only as a memory-safety defense.
     */
    uint64_t block_table;
    uint64_t host_k_base;
    uint64_t host_v_base;
    uint64_t device_k_base;
    uint64_t device_v_base;
    int64_t num_reqs;
    int64_t topk;
    int64_t capacity;
    int64_t max_token;
    int64_t max_num_blocks;
    int32_t block_size;
    int32_t token_size_bytes_k;
    int32_t token_size_bytes_v;
} sparse_kv_load_runtime_params_t;

/*
 * Plan-only ABI used by vLLM-Ascend fused sparse attention.  Logical inputs
 * and outputs are indexed by logical row; last_req_ids/slot_to_token/lru_slots
 * are persistent physical-row state.  The implementation first builds a
 * deterministic logical_to_physical map on device, then emits an int16 NPU
 * external plan consumed directly by the fused attention kernel on the same
 * stream.  No Host plan bridge and no TransferRuntime launch are involved.
 */
typedef struct sparse_kv_plan_fsa_runtime_params {
    uint64_t req_ids;
    uint64_t last_req_ids;
    uint64_t topk_indices;
    uint64_t stable_prefix_lens;
    uint64_t visible_seq_lens;
    uint64_t slot_to_token;
    uint64_t lru_slots;
    uint64_t current_slots;
    uint64_t miss_count;
    uint64_t miss_tokens;
    uint64_t miss_slots;
    uint64_t compact_workspace;
    uint64_t compact_workspace_bytes;
    uint64_t row_map_workspace;
    uint64_t row_map_workspace_bytes;
    uint64_t encoded_plan;
    uint64_t current_linear_slots;
    int64_t num_logical_rows;
    int64_t physical_row_capacity;
    int64_t topk;
    int64_t capacity;
    int64_t max_token;
    int64_t encoded_plan_stride;
} sparse_kv_plan_fsa_runtime_params_t;

enum {
    SPARSE_KV_FSA_PLAN_ALIGNMENT_INT16 = 16,
    SPARSE_KV_FSA_PLAN_CONTROL_COUNT_INT16 = 8,
    SPARSE_KV_FSA_PLAN_CONTROL_STORAGE_INT16 = 16,
};

#define SPARSE_KV_FSA_PLAN_READY_MARKER ((int16_t)0x5A4D)
#define SPARSE_KV_FSA_PLAN_EXTERNAL_READY_MARKER ((int16_t)0x5A45)
#define SPARSE_KV_FSA_PLAN_DIRECT_LAYOUT_MARKER ((int16_t)0x5A44)
#define SPARSE_KV_FSA_PLAN_PAIRED_COPY_MARKER ((int16_t)0x5A56)

static inline uint64_t sparse_kv_fsa_plan_control_offset_int16(int64_t topk)
{
    if (topk <= 0 || topk > INT32_MAX) {
        return 0;
    }
    uint64_t value = (uint64_t)topk;
    return (value + SPARSE_KV_FSA_PLAN_ALIGNMENT_INT16 - 1U) &
           ~(uint64_t)(SPARSE_KV_FSA_PLAN_ALIGNMENT_INT16 - 1U);
}

static inline uint64_t sparse_kv_fsa_plan_row_stride_int16(int64_t topk)
{
    uint64_t control_offset =
        sparse_kv_fsa_plan_control_offset_int16(topk);
    if (control_offset == 0 ||
        control_offset > UINT64_MAX -
            SPARSE_KV_FSA_PLAN_CONTROL_STORAGE_INT16) {
        return 0;
    }
    return control_offset + SPARSE_KV_FSA_PLAN_CONTROL_STORAGE_INT16;
}

/* logical_to_physical[num_logical_rows] + used[physical_row_capacity]. */
static inline uint64_t sparse_kv_fsa_row_map_workspace_size_bytes(
    int64_t num_logical_rows, int64_t physical_row_capacity)
{
    if (num_logical_rows <= 0 || physical_row_capacity < num_logical_rows) {
        return 0;
    }
    uint64_t logical = (uint64_t)num_logical_rows;
    uint64_t physical = (uint64_t)physical_row_capacity;
    if (logical > UINT64_MAX - physical ||
        logical + physical > UINT64_MAX / sizeof(int32_t)) {
        return 0;
    }
    return (logical + physical) * sizeof(int32_t);
}

/*
 * Runtime Plan workspace layout, in int32 elements, for every request row:
 *
 *   hash_keys[hash_capacity]
 *   hash_first_position[hash_capacity]
 *   hash_resident_owner[hash_capacity]
 *   evictable_slots[capacity]
 *   hit_slots[capacity]
 *   miss_positions[topk]
 *
 * hash_capacity is a power of two and keeps the load factor <= 0.5.  The
 * formula deliberately has no max_token term.
 */
static inline uint64_t sparse_kv_plan_hash_capacity(int64_t topk)
{
    if (topk <= 0 || topk > INT32_MAX / 2) {
        return 0;
    }
    uint64_t target = (uint64_t)topk * 2U;
    uint64_t capacity = 32U;
    while (capacity < target) {
        capacity <<= 1U;
    }
    return capacity;
}

/*
 * Return whether block_table has enough logical columns to cover every token
 * in [0, max_token).  Quotient/remainder ceil-division avoids the potentially
 * overflowing expression max_token + block_size - 1.
 */
static inline int sparse_kv_runtime_block_table_covers_tokens(
    int64_t max_token, int32_t block_size, int64_t max_num_blocks)
{
    if (max_token <= 0 || block_size <= 0 || max_num_blocks <= 0) {
        return 0;
    }
    uint64_t token_count = (uint64_t)max_token;
    uint64_t block_size_u64 = (uint64_t)block_size;
    uint64_t required_blocks =
        token_count / block_size_u64 +
        (uint64_t)((token_count % block_size_u64) != 0U);
    return (uint64_t)max_num_blocks >= required_blocks;
}

static inline uint64_t sparse_kv_plan_workspace_row_elements(
    int64_t topk, int64_t capacity)
{
    uint64_t hash_capacity = sparse_kv_plan_hash_capacity(topk);
    if (hash_capacity == 0 || capacity <= 0 || capacity > INT32_MAX) {
        return 0;
    }
    uint64_t cap = (uint64_t)capacity;
    uint64_t tk = (uint64_t)topk;
    if (hash_capacity > (UINT64_MAX - 2U * cap - tk) / 3U) {
        return 0;
    }
    return 3U * hash_capacity + 2U * cap + tk;
}

static inline uint64_t sparse_kv_plan_workspace_size_bytes(
    int64_t num_reqs, int64_t topk, int64_t capacity)
{
    uint64_t row_elements =
        sparse_kv_plan_workspace_row_elements(topk, capacity);
    if (row_elements == 0 || num_reqs <= 0) {
        return 0;
    }
    uint64_t rows = (uint64_t)num_reqs;
    if (rows > UINT64_MAX / row_elements / sizeof(int32_t)) {
        return 0;
    }
    return rows * row_elements * sizeof(int32_t);
}

#ifdef __cplusplus
}
#endif

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_SPARSE_KV_RUNTIME_H
