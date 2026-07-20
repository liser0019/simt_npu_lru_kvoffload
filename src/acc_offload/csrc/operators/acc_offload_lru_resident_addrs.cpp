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
#include "acc_offload_lru_resident_addrs.h"

extern "C" __global__ __aicore__ void OffloadComputeLruResidentAddrsOps(
    GM_ADDR miss_count, GM_ADDR miss_tokens, GM_ADDR miss_slots, GM_ADDR block_table, GM_ADDR gvas_buffer,
    GM_ADDR addr_buffer, GM_ADDR size_buffer, GM_ADDR num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base, int64_t gvas_v_base,
    int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs, int64_t topk,
    int64_t max_num_blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    ComputeLruResidentAddrsKernel op;
    op.Init(miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer, size_buffer,
            num_tokens_buffer, block_size, token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base,
            addr_k_base, addr_v_base, resident_capacity, num_reqs, topk, max_num_blocks);
    op.Process();
}

extern "C" void OffloadOpsComputeLruResidentAddrs(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots, uint64_t block_table, uint64_t gvas_buffer,
    uint64_t addr_buffer, uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base, int64_t gvas_v_base,
    int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs, int64_t topk,
    int64_t max_num_blocks, void *stream)
{
    constexpr uint32_t blockDim = LRU_ADDRS_BLOCK_DIM;
    OffloadComputeLruResidentAddrsOps<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(miss_count), reinterpret_cast<GM_ADDR>(miss_tokens),
        reinterpret_cast<GM_ADDR>(miss_slots), reinterpret_cast<GM_ADDR>(block_table),
        reinterpret_cast<GM_ADDR>(gvas_buffer), reinterpret_cast<GM_ADDR>(addr_buffer),
        reinterpret_cast<GM_ADDR>(size_buffer), reinterpret_cast<GM_ADDR>(num_tokens_buffer), block_size,
        token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks);
}
