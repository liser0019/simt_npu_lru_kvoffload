/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Internal launch interface for the production ComputeLruResidentAddrs mixed
 * SIMD/SIMT backend.  V2 supports runtime multi-request packing while keeping
 * the original single-request VF as a dedicated fast path.
 */

#ifndef ACC_OFFLOAD_LRU_RESIDENT_ADDRS_MIXED_PARALLEL_H
#define ACC_OFFLOAD_LRU_RESIDENT_ADDRS_MIXED_PARALLEL_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void OffloadOpsComputeLruResidentAddrsMixedParallel(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
    uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v,
    int64_t gvas_k_base, int64_t gvas_v_base, int64_t addr_k_base,
    int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs,
    int64_t topk, int64_t max_num_blocks, void *stream);

#ifdef __cplusplus
}
#endif

#endif // ACC_OFFLOAD_LRU_RESIDENT_ADDRS_MIXED_PARALLEL_H
