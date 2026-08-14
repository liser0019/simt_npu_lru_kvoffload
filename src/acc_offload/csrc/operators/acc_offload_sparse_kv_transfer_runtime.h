/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */
#ifndef ACC_OFFLOAD_SPARSE_KV_TRANSFER_RUNTIME_H
#define ACC_OFFLOAD_SPARSE_KV_TRANSFER_RUNTIME_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

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

#endif // ACC_OFFLOAD_SPARSE_KV_TRANSFER_RUNTIME_H
