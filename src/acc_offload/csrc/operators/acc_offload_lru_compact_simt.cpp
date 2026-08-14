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

#include <climits>
#include <cstdint>
#include <cstdio>
#include "simt_api/asc_simt.h"

namespace {
constexpr uint32_t LRU_COMPACT_MAX_GRID_DIM = 8;
constexpr uint32_t LRU_COMPACT_MAX_THREADS_PER_BLOCK = 128;
constexpr int64_t LRU_COMPACT_WORKSPACE_THREADS = 8;
constexpr int32_t LRU_EPOCH_RESET_THRESHOLD = 1 << 30;
constexpr int32_t LRU_INVALID_TOKEN = -1;
}

// The production implementation is split into same-stream phases.  This gives
// the row one 128-thread block for independent work without relying on an
// undocumented CUDA-style intra-block barrier spelling.  The only single-thread
// phase is order-sensitive and linear in max_token + topk + capacity.
extern "C" __global__ void OffloadLruCompactResetPhasedSimtOps(
    int64_t *req_ids, int64_t *last_req_ids, int32_t *slot_to_token, int32_t *lru_slots,
    int64_t row, int64_t capacity)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || row < 0) {
        return;
    }
    if (last_req_ids[row] == req_ids[row]) {
        return;
    }

    int64_t slot_base = row * capacity;
    for (int64_t slot = 0; slot < capacity; ++slot) {
        slot_to_token[slot_base + slot] = LRU_INVALID_TOKEN;
        lru_slots[slot_base + slot] = static_cast<int32_t>(slot);
    }
    last_req_ids[row] = req_ids[row];
}

extern "C" __global__ void OffloadLruCompactPreparePhasedSimtOps(
    int32_t *stable_prefix_lens, int32_t *slot_to_token, int32_t *current_slots,
    int32_t *miss_count, int32_t *miss_tokens, int32_t *miss_slots,
    int64_t row, int64_t topk, int64_t capacity, int64_t max_token)
{
    int64_t thread = static_cast<int64_t>(threadIdx.x);
    int64_t topk_base = row * topk;
    int64_t slot_base = row * capacity;

    for (int64_t pos = thread; pos < topk; pos += blockDim.x) {
        current_slots[topk_base + pos] = LRU_INVALID_TOKEN;
        miss_tokens[topk_base + pos] = LRU_INVALID_TOKEN;
        miss_slots[topk_base + pos] = LRU_INVALID_TOKEN;
    }
    if (thread == 0) {
        miss_count[row] = 0;
    }

    int64_t stable_prefix_len = stable_prefix_lens[row];
    if (stable_prefix_len < 0) {
        stable_prefix_len = 0;
    } else if (stable_prefix_len > max_token) {
        stable_prefix_len = max_token;
    }
    for (int64_t slot = thread; slot < capacity; slot += blockDim.x) {
        int32_t token = slot_to_token[slot_base + slot];
        if (token >= 0 && token < max_token && token >= stable_prefix_len) {
            slot_to_token[slot_base + slot] = LRU_INVALID_TOKEN;
        }
    }
}

// One ordering thread constructs a deterministic linear plan.  Negative epoch
// values cannot collide with the non-negative slot IDs stored in token_mark as
// scratch after classification.  token_pos[max_token-2:max_token] stores the
// two plan counts; the optimized Host path validates topk <= max_token - 2.
extern "C" __global__ void OffloadLruCompactSimtOps(
    int32_t *topk_indices, int32_t *slot_to_token, int32_t *lru_slots,
    int32_t *current_slots, int32_t *miss_count, int32_t *miss_tokens,
    int32_t *token_mark, int32_t *token_pos, int32_t *epoch,
    int64_t row, int64_t topk, int64_t capacity, int64_t max_token)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int64_t topk_base = row * topk;
    int64_t slot_base = row * capacity;
    int32_t previous_epoch = *epoch;
    int32_t current_epoch = -2;
    if (previous_epoch >= -1 || previous_epoch <= 1 - LRU_EPOCH_RESET_THRESHOLD) {
        for (int64_t token = 0; token < max_token; ++token) {
            token_mark[token] = 0;
            token_pos[token] = LRU_INVALID_TOKEN;
        }
    } else {
        current_epoch = previous_epoch - 1;
    }
    *epoch = current_epoch;

    // Preserve the CPU/AIV compatibility rule: token_pos records the first
    // Top-K occurrence.  Later duplicate positions remain independent misses.
    for (int64_t pos = 0; pos < topk; ++pos) {
        int32_t token = topk_indices[topk_base + pos];
        if (token >= 0 && token < max_token && token_mark[token] != current_epoch) {
            token_mark[token] = current_epoch;
            token_pos[token] = static_cast<int32_t>(pos);
        }
    }

    // Encode the hit predicate in-place in the old LRU row.  The subsequent
    // scan no longer needs token membership, allowing token_mark to become a
    // capacity-sized stable slot scratch array without another public ABI.
    for (int64_t order = 0; order < capacity; ++order) {
        int32_t slot = lru_slots[slot_base + order];
        if (slot < 0 || slot >= capacity) {
            lru_slots[slot_base + order] = LRU_INVALID_TOKEN;
            continue;
        }
        int32_t token = slot_to_token[slot_base + slot];
        if (token >= 0 && token < max_token && token_mark[token] == current_epoch) {
            int32_t pos = token_pos[token];
            current_slots[topk_base + pos] = slot;
            lru_slots[slot_base + order] = -slot - 2;
        } else {
            lru_slots[slot_base + order] = slot;
        }
    }

    int32_t evictable_count = 0;
    int32_t hit_count = 0;
    for (int64_t order = 0; order < capacity; ++order) {
        int32_t encoded = lru_slots[slot_base + order];
        if (encoded <= -2) {
            int32_t slot = -encoded - 2;
            token_mark[max_token - 1 - hit_count] = slot;
            ++hit_count;
        } else if (encoded >= 0 && encoded < capacity) {
            token_mark[evictable_count] = encoded;
            ++evictable_count;
        }
    }

    int32_t local_miss_count = 0;
    for (int64_t pos = 0; pos < topk; ++pos) {
        int32_t token = topk_indices[topk_base + pos];
        if (token >= 0 && token < max_token && current_slots[topk_base + pos] < 0) {
            miss_tokens[topk_base + local_miss_count] = token;
            token_pos[local_miss_count] = static_cast<int32_t>(pos);
            ++local_miss_count;
        }
    }

    int32_t assign_count = local_miss_count < evictable_count ? local_miss_count : evictable_count;
    miss_count[row] = assign_count;
    token_pos[max_token - 2] = evictable_count;
    token_pos[max_token - 1] = hit_count;
}

extern "C" __global__ void OffloadLruCompactAssignPhasedSimtOps(
    int32_t *slot_to_token, int32_t *current_slots, int32_t *miss_count,
    int32_t *miss_tokens, int32_t *miss_slots, int32_t *token_mark,
    int32_t *token_pos, int64_t row, int64_t topk, int64_t capacity)
{
    int64_t topk_base = row * topk;
    int32_t assign_count = miss_count[row];
    for (int32_t miss_idx = static_cast<int32_t>(threadIdx.x); miss_idx < assign_count;
         miss_idx += static_cast<int32_t>(blockDim.x)) {
        int32_t slot = token_mark[miss_idx];
        int32_t token = miss_tokens[topk_base + miss_idx];
        int32_t pos = token_pos[miss_idx];
        if (slot >= 0 && slot < capacity && pos >= 0 && pos < topk) {
            slot_to_token[row * capacity + slot] = token;
            current_slots[topk_base + pos] = slot;
            miss_slots[topk_base + miss_idx] = slot;
        }
    }
}

extern "C" __global__ void OffloadLruCompactRebuildPhasedSimtOps(
    int32_t *lru_slots, int32_t *miss_count, int32_t *miss_slots,
    int32_t *token_mark, int32_t *token_pos,
    int64_t row, int64_t topk, int64_t capacity, int64_t max_token)
{
    int32_t assign_count = miss_count[row];
    int32_t evictable_count = token_pos[max_token - 2];
    int32_t hit_count = token_pos[max_token - 1];
    int32_t unassigned_count = evictable_count - assign_count;
    int32_t valid_count = evictable_count + hit_count;
    int64_t slot_base = row * capacity;
    int64_t topk_base = row * topk;

    for (int64_t output = static_cast<int64_t>(threadIdx.x); output < capacity;
         output += blockDim.x) {
        int32_t slot = LRU_INVALID_TOKEN;
        if (output < unassigned_count) {
            slot = token_mark[assign_count + output];
        } else if (output < evictable_count) {
            slot = miss_slots[topk_base + output - unassigned_count];
        } else if (output < valid_count) {
            int32_t hit_idx = static_cast<int32_t>(output) - evictable_count;
            slot = token_mark[max_token - 1 - hit_idx];
        }
        lru_slots[slot_base + output] = slot;
    }
}

// Legacy colleague implementation retained as a distinct device symbol for
// rollback and source-level A/B.  Its one-thread-per-row complexity is not used
// for production-supported optimized shapes.
// Each SIMT thread owns one request row. The algorithm intentionally uses GM
// scalar loads/stores: LRU lookup and compaction are irregular and branch-heavy,
// which is the workload targeted by A5 SIMT. token_mark_workspace is reused as
// per-thread compaction scratch; token_pos_workspace and epochs remain in the
// ABI for compatibility with the AIV implementation.
extern "C" __global__ void OffloadLruCompactLegacySimtOps(
    int64_t *req_ids, int64_t *last_req_ids, int32_t *topk_indices, int32_t *stable_prefix_lens,
    int32_t *slot_to_token, int32_t *lru_slots, int32_t *current_slots, int32_t *miss_count,
    int32_t *miss_tokens, int32_t *miss_slots, int32_t *token_mark_workspace, int32_t *token_pos_workspace,
    int32_t *epochs, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token)
{
    (void)token_pos_workspace;
    (void)epochs;

    int64_t row_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t row = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(threadIdx.x) * gridDim.x;

    if (topk <= 0 || capacity <= 0 || max_token <= 0) {
        for (; row < num_reqs; row += row_stride) {
            miss_count[row] = 0;
        }
        return;
    }

    for (; row < num_reqs; row += row_stride) {
        int64_t topk_base = row * topk;
        int64_t slot_base = row * capacity;
        bool has_scratch = capacity <= max_token;
        int32_t *evictable_scratch = nullptr;
        if (has_scratch) {
            evictable_scratch = token_mark_workspace + static_cast<int64_t>(blockIdx.x) * max_token +
                                static_cast<int64_t>(threadIdx.x) * capacity;
        }

        for (int64_t pos = 0; pos < topk; ++pos) {
            current_slots[topk_base + pos] = LRU_INVALID_TOKEN;
            miss_tokens[topk_base + pos] = LRU_INVALID_TOKEN;
            miss_slots[topk_base + pos] = LRU_INVALID_TOKEN;
        }

        int64_t req_id = req_ids[row];
        if (last_req_ids[row] != req_id) {
            for (int64_t slot = 0; slot < capacity; ++slot) {
                slot_to_token[slot_base + slot] = LRU_INVALID_TOKEN;
                lru_slots[slot_base + slot] = static_cast<int32_t>(slot);
            }
            last_req_ids[row] = req_id;
        }

        int64_t stable_prefix_len = stable_prefix_lens[row];
        if (stable_prefix_len < 0) {
            stable_prefix_len = 0;
        } else if (stable_prefix_len > max_token) {
            stable_prefix_len = max_token;
        }

        // Invalidate cached speculative-suffix entries before hit detection.
        for (int64_t slot = 0; slot < capacity; ++slot) {
            int32_t token = slot_to_token[slot_base + slot];
            if (token >= 0 && token < max_token && token >= stable_prefix_len) {
                slot_to_token[slot_base + slot] = LRU_INVALID_TOKEN;
            }
        }

        // Scan in old LRU order, resolve all top-k hits, and retain evictable
        // slots in per-thread scratch when the caller's workspace permits it.
        int32_t evictable_count = 0;
        for (int64_t order = 0; order < capacity; ++order) {
            int32_t slot = lru_slots[slot_base + order];
            if (slot < 0 || slot >= capacity) {
                continue;
            }
            int32_t resident_token = slot_to_token[slot_base + slot];
            bool hit = false;
            if (resident_token >= 0 && resident_token < max_token) {
                for (int64_t pos = 0; pos < topk; ++pos) {
                    if (topk_indices[topk_base + pos] == resident_token) {
                        hit = true;
                        current_slots[topk_base + pos] = slot;
                    }
                }
            }
            if (!hit) {
                if (has_scratch) {
                    evictable_scratch[evictable_count] = slot;
                }
                ++evictable_count;
            }
        }

        // Collect unique misses. miss_slots is temporarily used to retain each
        // miss's top-k position until a resident slot has been assigned.
        int32_t local_miss_count = 0;
        for (int64_t pos = 0; pos < topk; ++pos) {
            int32_t token = topk_indices[topk_base + pos];
            if (token < 0 || token >= max_token || current_slots[topk_base + pos] >= 0) {
                continue;
            }
            bool duplicate = false;
            for (int64_t prior = 0; prior < pos; ++prior) {
                if (topk_indices[topk_base + prior] == token) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                miss_tokens[topk_base + local_miss_count] = token;
                miss_slots[topk_base + local_miss_count] = static_cast<int32_t>(pos);
                ++local_miss_count;
            }
        }

        // Allocate the oldest slot that is not protected by a top-k hit or by a
        // previously assigned miss.
        int32_t assign_count = 0;
        for (int32_t miss_idx = 0; miss_idx < local_miss_count; ++miss_idx) {
            int32_t chosen_slot = LRU_INVALID_TOKEN;
            if (has_scratch) {
                if (miss_idx < evictable_count) {
                    chosen_slot = evictable_scratch[miss_idx];
                }
            } else {
                for (int64_t order = 0; order < capacity; ++order) {
                    int32_t candidate = lru_slots[slot_base + order];
                    if (candidate < 0 || candidate >= capacity) {
                        continue;
                    }
                    bool protected_slot = false;
                    for (int64_t pos = 0; pos < topk; ++pos) {
                        if (current_slots[topk_base + pos] == candidate) {
                            protected_slot = true;
                            break;
                        }
                    }
                    if (!protected_slot) {
                        chosen_slot = candidate;
                        break;
                    }
                }
            }
            if (chosen_slot < 0) {
                break;
            }

            int32_t token = miss_tokens[topk_base + miss_idx];
            int32_t pos = miss_slots[topk_base + miss_idx];
            current_slots[topk_base + pos] = chosen_slot;
            miss_slots[topk_base + miss_idx] = chosen_slot;
            ++assign_count;
        }

        // Propagate the resolved slot to duplicate top-k positions.
        for (int64_t pos = 0; pos < topk; ++pos) {
            if (current_slots[topk_base + pos] >= 0) {
                continue;
            }
            int32_t token = topk_indices[topk_base + pos];
            for (int64_t prior = 0; prior < pos; ++prior) {
                if (topk_indices[topk_base + prior] == token && current_slots[topk_base + prior] >= 0) {
                    current_slots[topk_base + pos] = current_slots[topk_base + prior];
                    break;
                }
            }
        }

        for (int64_t i = assign_count; i < topk; ++i) {
            miss_tokens[topk_base + i] = LRU_INVALID_TOKEN;
            miss_slots[topk_base + i] = LRU_INVALID_TOKEN;
        }

        if (has_scratch) {
            // Write hit slots to the tail while scanning the old LRU order in
            // reverse. The destination is never before the unread source.
            int64_t write_pos = capacity - 1;
            for (int64_t order = capacity - 1; order >= 0; --order) {
                int32_t slot = lru_slots[slot_base + order];
                if (slot < 0 || slot >= capacity) {
                    continue;
                }
                int32_t token = slot_to_token[slot_base + slot];
                bool hit = false;
                for (int64_t pos = 0; pos < topk; ++pos) {
                    if (token >= 0 && token == topk_indices[topk_base + pos]) {
                        hit = true;
                        break;
                    }
                }
                if (hit) {
                    lru_slots[slot_base + write_pos] = slot;
                    --write_pos;
                }
            }
            int64_t output = 0;
            for (int32_t i = assign_count; i < evictable_count; ++i) {
                lru_slots[slot_base + output++] = evictable_scratch[i];
            }
            for (int32_t i = 0; i < assign_count; ++i) {
                lru_slots[slot_base + output++] = miss_slots[topk_base + i];
            }
        } else {
            // Defensive fallback for the nonsensical capacity > max_token case,
            // where the documented workspace cannot hold one row of scratch.
            for (int64_t i = 1; i < capacity; ++i) {
                int32_t key_slot = lru_slots[slot_base + i];
                int32_t key_category = 0;
                for (int32_t m = 0; m < assign_count; ++m) {
                    if (miss_slots[topk_base + m] == key_slot) {
                        key_category = 1;
                        break;
                    }
                }
                if (key_category == 0) {
                    for (int64_t pos = 0; pos < topk; ++pos) {
                        if (current_slots[topk_base + pos] == key_slot) {
                            key_category = 2;
                            break;
                        }
                    }
                }
                int64_t j = i - 1;
                while (j >= 0) {
                    int32_t prior_slot = lru_slots[slot_base + j];
                    int32_t prior_category = 0;
                    for (int32_t m = 0; m < assign_count; ++m) {
                        if (miss_slots[topk_base + m] == prior_slot) {
                            prior_category = 1;
                            break;
                        }
                    }
                    if (prior_category == 0) {
                        for (int64_t pos = 0; pos < topk; ++pos) {
                            if (current_slots[topk_base + pos] == prior_slot) {
                                prior_category = 2;
                                break;
                            }
                        }
                    }
                    if (prior_category <= key_category) {
                        break;
                    }
                    lru_slots[slot_base + j + 1] = prior_slot;
                    --j;
                }
                lru_slots[slot_base + j + 1] = key_slot;
            }
        }

        for (int32_t i = 0; i < assign_count; ++i) {
            slot_to_token[slot_base + miss_slots[topk_base + i]] = miss_tokens[topk_base + i];
        }

        miss_count[row] = assign_count;
    }
}

namespace {
void LaunchLegacyLruCompact(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices, uint64_t stable_prefix_lens,
    uint64_t slot_to_token, uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
    uint64_t miss_tokens, uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
    uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token, void *stream)
{
    uint32_t grid_dim = 1;
    if (num_reqs > LRU_COMPACT_MAX_GRID_DIM) {
        grid_dim = LRU_COMPACT_MAX_GRID_DIM;
    } else if (num_reqs > 0) {
        grid_dim = static_cast<uint32_t>(num_reqs);
    }
    uint32_t threads_per_block = 1;
    if (capacity > 0 && max_token >= capacity) {
        int64_t workspace_threads = max_token / capacity;
        threads_per_block = static_cast<uint32_t>(workspace_threads > LRU_COMPACT_MAX_THREADS_PER_BLOCK
                                                      ? LRU_COMPACT_MAX_THREADS_PER_BLOCK
                                                      : workspace_threads);
    }
    OffloadLruCompactLegacySimtOps<<<grid_dim, threads_per_block, 0, stream>>>(
        reinterpret_cast<int64_t *>(req_ids), reinterpret_cast<int64_t *>(last_req_ids),
        reinterpret_cast<int32_t *>(topk_indices), reinterpret_cast<int32_t *>(stable_prefix_lens),
        reinterpret_cast<int32_t *>(slot_to_token), reinterpret_cast<int32_t *>(lru_slots),
        reinterpret_cast<int32_t *>(current_slots), reinterpret_cast<int32_t *>(miss_count),
        reinterpret_cast<int32_t *>(miss_tokens), reinterpret_cast<int32_t *>(miss_slots),
        reinterpret_cast<int32_t *>(token_mark_workspace), reinterpret_cast<int32_t *>(token_pos_workspace),
        reinterpret_cast<int32_t *>(epochs), num_reqs, topk, capacity, max_token);
}

bool CanUseProductionPhasedLru(int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token)
{
    return num_reqs > 0 && topk > 0 && capacity > 0 && max_token >= 3 &&
           capacity <= max_token && topk <= max_token - 2 &&
           topk < static_cast<int64_t>(INT32_MAX) &&
           capacity < static_cast<int64_t>(INT32_MAX) &&
           max_token <= static_cast<int64_t>(INT32_MAX) &&
           num_reqs <= INT64_MAX / topk && num_reqs <= INT64_MAX / capacity &&
           max_token <= INT64_MAX / LRU_COMPACT_WORKSPACE_THREADS;
}

void LaunchProductionPhasedLru(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices, uint64_t stable_prefix_lens,
    uint64_t slot_to_token, uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
    uint64_t miss_tokens, uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
    uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token, void *stream)
{
    auto *req_ids_ptr = reinterpret_cast<int64_t *>(req_ids);
    auto *last_req_ids_ptr = reinterpret_cast<int64_t *>(last_req_ids);
    auto *topk_indices_ptr = reinterpret_cast<int32_t *>(topk_indices);
    auto *stable_prefix_lens_ptr = reinterpret_cast<int32_t *>(stable_prefix_lens);
    auto *slot_to_token_ptr = reinterpret_cast<int32_t *>(slot_to_token);
    auto *lru_slots_ptr = reinterpret_cast<int32_t *>(lru_slots);
    auto *current_slots_ptr = reinterpret_cast<int32_t *>(current_slots);
    auto *miss_count_ptr = reinterpret_cast<int32_t *>(miss_count);
    auto *miss_tokens_ptr = reinterpret_cast<int32_t *>(miss_tokens);
    auto *miss_slots_ptr = reinterpret_cast<int32_t *>(miss_slots);
    auto *token_mark_ptr = reinterpret_cast<int32_t *>(token_mark_workspace);
    auto *token_pos_ptr = reinterpret_cast<int32_t *>(token_pos_workspace);
    auto *epochs_ptr = reinterpret_cast<int32_t *>(epochs);

    // Rows are deliberately submitted one at a time.  This keeps adjacent GM
    // rows from being dirtied by separate private SIMT DCaches and lets the
    // existing eight workspace slices be reused without a new public ABI.
    for (int64_t row = 0; row < num_reqs; ++row) {
        int64_t workspace_index = row % LRU_COMPACT_WORKSPACE_THREADS;
        int32_t *mark_row = token_mark_ptr + workspace_index * max_token;
        int32_t *pos_row = token_pos_ptr + workspace_index * max_token;
        int32_t *epoch = epochs_ptr + workspace_index;

        // NEEDS_SERVER_CONFIRMATION: CANN 9.1/dav-3510 must preserve direct
        // SIMT GM/DCache visibility across these same-stream kernel boundaries.
        OffloadLruCompactResetPhasedSimtOps<<<1, 1, 0, stream>>>(
            req_ids_ptr, last_req_ids_ptr, slot_to_token_ptr, lru_slots_ptr, row, capacity);
        OffloadLruCompactPreparePhasedSimtOps<<<1, LRU_COMPACT_MAX_THREADS_PER_BLOCK, 0, stream>>>(
            stable_prefix_lens_ptr, slot_to_token_ptr, current_slots_ptr, miss_count_ptr,
            miss_tokens_ptr, miss_slots_ptr, row, topk, capacity, max_token);
        OffloadLruCompactSimtOps<<<1, 1, 0, stream>>>(
            topk_indices_ptr, slot_to_token_ptr, lru_slots_ptr, current_slots_ptr,
            miss_count_ptr, miss_tokens_ptr, mark_row, pos_row, epoch,
            row, topk, capacity, max_token);
        OffloadLruCompactAssignPhasedSimtOps<<<1, LRU_COMPACT_MAX_THREADS_PER_BLOCK, 0, stream>>>(
            slot_to_token_ptr, current_slots_ptr, miss_count_ptr, miss_tokens_ptr,
            miss_slots_ptr, mark_row, pos_row, row, topk, capacity);
        OffloadLruCompactRebuildPhasedSimtOps<<<1, LRU_COMPACT_MAX_THREADS_PER_BLOCK, 0, stream>>>(
            lru_slots_ptr, miss_count_ptr, miss_slots_ptr, mark_row, pos_row,
            row, topk, capacity, max_token);
    }
}
} // namespace

extern "C" void OffloadOpsLruCompact(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices, uint64_t stable_prefix_lens,
    uint64_t slot_to_token, uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
    uint64_t miss_tokens, uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
    uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token, void *stream)
{
    if (!CanUseProductionPhasedLru(num_reqs, topk, capacity, max_token)) {
        std::fprintf(stderr,
            "[SIMT_LRU_FALLBACK] using legacy thread-per-row Compact: "
            "num_reqs=%lld topk=%lld capacity=%lld max_token=%lld\n",
            static_cast<long long>(num_reqs), static_cast<long long>(topk),
            static_cast<long long>(capacity), static_cast<long long>(max_token));
        std::fflush(stderr);
        LaunchLegacyLruCompact(req_ids, last_req_ids, topk_indices, stable_prefix_lens,
            slot_to_token, lru_slots, current_slots, miss_count, miss_tokens, miss_slots,
            token_mark_workspace, token_pos_workspace, epochs, num_reqs, topk,
            capacity, max_token, stream);
        return;
    }
    LaunchProductionPhasedLru(req_ids, last_req_ids, topk_indices, stable_prefix_lens,
        slot_to_token, lru_slots, current_slots, miss_count, miss_tokens, miss_slots,
        token_mark_workspace, token_pos_workspace, epochs, num_reqs, topk,
        capacity, max_token, stream);
}
