/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Internal device-launch interface for the production Compact V3 kernel.
 *
 * This header deliberately exposes one optimized implementation only.  The
 * public Python/C++ API remains AccOffloadLruCompact; the Host launcher uses
 * this symbol for the fixed production shape and transparently falls back to
 * the generic SIMT implementation for other shapes.
 */

#ifndef ACC_OFFLOAD_LRU_COMPACT_FUSED_V3_H
#define ACC_OFFLOAD_LRU_COMPACT_FUSED_V3_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void OffloadOpsLruCompactFusedV3(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
    uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
    uint64_t miss_slots, uint64_t token_mark_workspace,
    uint64_t token_pos_workspace, uint64_t epochs, int64_t row,
    int64_t topk, int64_t capacity, int64_t max_token, void *stream,
    uint32_t profile_enabled);

#ifdef __cplusplus
}
#endif

#endif // ACC_OFFLOAD_LRU_COMPACT_FUSED_V3_H
