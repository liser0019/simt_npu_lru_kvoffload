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

#include <cstdint>
#include "simt_api/asc_simt.h"

namespace {
constexpr uint32_t LRU_ADDRS_MAX_GRID_DIM = 8;
constexpr uint32_t LRU_ADDRS_THREADS_PER_BLOCK = 128;
}

extern "C" __global__ void OffloadComputeLruResidentAddrsSimtOps(
    int32_t *miss_count, int32_t *miss_tokens, int32_t *miss_slots, int32_t *block_table,
    int64_t *gvas_buffer, int64_t *addr_buffer, int32_t *size_buffer, int32_t *num_tokens_buffer,
    int32_t block_size, int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base,
    int64_t gvas_v_base, int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity,
    int64_t num_reqs, int64_t topk, int64_t max_num_blocks)
{
    int64_t row_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t req = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(threadIdx.x) * gridDim.x;

    if (num_reqs <= 0 || block_size <= 0 || resident_capacity <= 0 || topk <= 0 || max_num_blocks <= 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *num_tokens_buffer = 0;
        }
        return;
    }
    if (req >= num_reqs) {
        return;
    }

    // Compute a robust packed layout. Invalid descriptors are excluded from the
    // prefix sum, so sparse-copy never receives an uninitialized address.
    int64_t total_valid = 0;
    int64_t req_start = 0;
    for (int64_t r = 0; r < num_reqs; ++r) {
        int32_t count = miss_count[r];
        if (count < 0) {
            count = 0;
        } else if (count > topk) {
            count = static_cast<int32_t>(topk);
        }

        int32_t valid_count = 0;
        for (int32_t i = 0; i < count; ++i) {
            int64_t offset = r * topk + i;
            int32_t token = miss_tokens[offset];
            int32_t slot = miss_slots[offset];
            if (token < 0 || slot < 0 || slot >= resident_capacity) {
                continue;
            }
            int64_t block_id = token / block_size;
            if (block_id < 0 || block_id >= max_num_blocks) {
                continue;
            }
            int32_t block_index = block_table[r * max_num_blocks + block_id];
            if (block_index < 0) {
                continue;
            }
            ++valid_count;
        }
        if (r < req) {
            req_start += valid_count;
        }
        total_valid += valid_count;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *num_tokens_buffer = static_cast<int32_t>(total_valid * 2);
    }

    int64_t block_bytes_k = static_cast<int64_t>(block_size) * token_size_bytes_k;
    int64_t block_bytes_v = static_cast<int64_t>(block_size) * token_size_bytes_v;

    for (; req < num_reqs; req += row_stride) {
        int32_t count = miss_count[req];
        if (count < 0) {
            count = 0;
        } else if (count > topk) {
            count = static_cast<int32_t>(topk);
        }

        int64_t packed_index = req_start;
        for (int32_t i = 0; i < count; ++i) {
            int64_t input_offset = req * topk + i;
            int32_t token = miss_tokens[input_offset];
            int32_t slot = miss_slots[input_offset];
            if (token < 0 || slot < 0 || slot >= resident_capacity) {
                continue;
            }

            int64_t block_id = token / block_size;
            if (block_id < 0 || block_id >= max_num_blocks) {
                continue;
            }
            int32_t block_index = block_table[req * max_num_blocks + block_id];
            if (block_index < 0) {
                continue;
            }

            int64_t offset_in_block = token % block_size;
            int64_t k_pos = packed_index;
            int64_t v_pos = total_valid + packed_index;

            gvas_buffer[k_pos] = gvas_k_base + static_cast<int64_t>(block_index) * block_bytes_k +
                                 offset_in_block * token_size_bytes_k;
            gvas_buffer[v_pos] = gvas_v_base + static_cast<int64_t>(block_index) * block_bytes_v +
                                 offset_in_block * token_size_bytes_v;
            addr_buffer[k_pos] = addr_k_base + (req * resident_capacity + slot) * token_size_bytes_k;
            addr_buffer[v_pos] = addr_v_base + (req * resident_capacity + slot) * token_size_bytes_v;
            size_buffer[k_pos] = token_size_bytes_k;
            size_buffer[v_pos] = token_size_bytes_v;
            ++packed_index;
        }

        // Advance to the packed start of the next row owned by this thread.
        int64_t next_req = req + row_stride;
        if (next_req < num_reqs) {
            req_start = 0;
            for (int64_t r = 0; r < next_req; ++r) {
                int32_t prior_count = miss_count[r];
                if (prior_count < 0) {
                    prior_count = 0;
                } else if (prior_count > topk) {
                    prior_count = static_cast<int32_t>(topk);
                }
                for (int32_t i = 0; i < prior_count; ++i) {
                    int64_t offset = r * topk + i;
                    int32_t token = miss_tokens[offset];
                    int32_t slot = miss_slots[offset];
                    if (token < 0 || slot < 0 || slot >= resident_capacity) {
                        continue;
                    }
                    int64_t block_id = token / block_size;
                    if (block_id >= 0 && block_id < max_num_blocks &&
                        block_table[r * max_num_blocks + block_id] >= 0) {
                        ++req_start;
                    }
                }
            }
        }
    }
}

extern "C" void OffloadOpsComputeLruResidentAddrs(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots, uint64_t block_table, uint64_t gvas_buffer,
    uint64_t addr_buffer, uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base, int64_t gvas_v_base,
    int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs, int64_t topk,
    int64_t max_num_blocks, void *stream)
{
    uint32_t grid_dim = 1;
    if (num_reqs > LRU_ADDRS_MAX_GRID_DIM) {
        grid_dim = LRU_ADDRS_MAX_GRID_DIM;
    } else if (num_reqs > 0) {
        grid_dim = static_cast<uint32_t>(num_reqs);
    }
    OffloadComputeLruResidentAddrsSimtOps<<<grid_dim, LRU_ADDRS_THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<int32_t *>(miss_count), reinterpret_cast<int32_t *>(miss_tokens),
        reinterpret_cast<int32_t *>(miss_slots), reinterpret_cast<int32_t *>(block_table),
        reinterpret_cast<int64_t *>(gvas_buffer), reinterpret_cast<int64_t *>(addr_buffer),
        reinterpret_cast<int32_t *>(size_buffer), reinterpret_cast<int32_t *>(num_tokens_buffer), block_size,
        token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks);
}
