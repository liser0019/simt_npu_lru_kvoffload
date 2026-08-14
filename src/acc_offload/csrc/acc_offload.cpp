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
