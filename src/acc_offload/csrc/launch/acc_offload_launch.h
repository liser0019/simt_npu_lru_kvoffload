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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H

#include <mutex>
#include "acc_offload_define.h"

namespace ock {
namespace offload {

using AccOffloadSparseCopyFunc = void (*)(uint64_t *, uint64_t *, uint32_t *, uint32_t *, uint8_t);

using AccOffloadLruCompactFunc = void (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int64_t, int64_t,
                                          int64_t, int64_t, uint8_t);

using AccOffloadComputeLruResidentAddrsFunc = void (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                                       uint64_t, uint64_t, int32_t, int32_t, int32_t, int64_t, int64_t,
                                                       int64_t, int64_t, int32_t, int64_t, int64_t, int64_t, uint8_t);

class AccOffloadLaunchApi {
public:
    static int32_t TryLoadLibrary();
    static void CleanupLibrary();

    static inline int32_t AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                               uint32_t *sizePtr, uint8_t devIdx)
    {
        if (pAccOffloadSparseCopy == nullptr) {
            return OFFLOAD_UNLOAD;
        }

        pAccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx);
        return OFFLOAD_OK;
    }

    static inline int32_t AccOffloadLruCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                               uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
                                               uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
                                               uint64_t miss_slots, uint64_t token_mark_workspace,
                                               uint64_t token_pos_workspace, uint64_t epochs, int64_t num_reqs,
                                               int64_t topk, int64_t capacity, int64_t max_token, uint8_t devIdx)
    {
        if (pAccOffloadLruCompact == nullptr) {
            return OFFLOAD_UNLOAD;
        }

        pAccOffloadLruCompact(req_ids, last_req_ids, topk_indices, stable_prefix_lens, slot_to_token, lru_slots,
                              current_slots, miss_count, miss_tokens, miss_slots, token_mark_workspace,
                              token_pos_workspace, epochs, num_reqs, topk, capacity, max_token, devIdx);
        return OFFLOAD_OK;
    }

    static inline int32_t AccOffloadComputeLruResidentAddrs(
        uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots, uint64_t block_table, uint64_t gvas_buffer,
        uint64_t addr_buffer, uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
        int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base, int64_t gvas_v_base,
        int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs, int64_t topk,
        int64_t max_num_blocks, uint8_t devIdx)
    {
        if (pAccOffloadComputeLruResidentAddrs == nullptr) {
            return OFFLOAD_UNLOAD;
        }

        pAccOffloadComputeLruResidentAddrs(miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer,
                                           size_buffer, num_tokens_buffer, block_size, token_size_bytes_k,
                                           token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
                                           resident_capacity, num_reqs, topk, max_num_blocks, devIdx);
        return OFFLOAD_OK;
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *libHandle;
    static const char *gAccOffloadLibName;

    static AccOffloadSparseCopyFunc pAccOffloadSparseCopy;
    static AccOffloadLruCompactFunc pAccOffloadLruCompact;
    static AccOffloadComputeLruResidentAddrsFunc pAccOffloadComputeLruResidentAddrs;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H