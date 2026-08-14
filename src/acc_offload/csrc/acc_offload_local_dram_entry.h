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
#ifndef MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H
#define MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H

#include <mutex>
#include <memory>
#include "hybm_def.h"
#include "acc_offload.h"
#include "acc_offload_entry.h"
#include "acc_offload_mem_manager.h"

namespace ock {
namespace offload {

class AccOffloadLocalDramEntry : public AccOffloadEntry {
public:
    AccOffloadLocalDramEntry() = default;
    ~AccOffloadLocalDramEntry() override
    {
        UnInitalize();
    };

    AccOffloadLocalDramEntry(const AccOffloadLocalDramEntry &) = delete;
    AccOffloadLocalDramEntry &operator=(const AccOffloadLocalDramEntry &) = delete;

    int32_t Initialize(const offload_config_t &config) override;

    void UnInitalize() override;

public:
    void *MallocHost(size_t size) override;

    void FreeHost(void *ptr) override;

    uint64_t GetDeviceAddress(const void *ptr, size_t size) override;

    int32_t SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                       uint8_t devIdx) override;

    int32_t LruResidentCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                               uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
                               uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
                               uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                               uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token,
                               uint8_t devIdx) override;

    int32_t ComputeLruResidentAddrs(uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                    uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
                                    uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
                                    int32_t token_size_bytes_k, int32_t token_size_bytes_v, int64_t gvas_k_base,
                                    int64_t gvas_v_base, int64_t addr_k_base, int64_t addr_v_base,
                                    int32_t resident_capacity, int64_t num_reqs, int64_t topk, int64_t max_num_blocks,
                                    uint8_t devIdx) override;

    int32_t SparseKvLoadRuntime(
        const sparse_kv_load_runtime_params_t &params,
        uint8_t devIdx) override;

private:
    std::mutex mutex_;
    bool inited_ = false;
    bool registerHostMemory_ = false;
    hybm_entity_t entity_ = nullptr;
    hybm_mem_slice_t slice_ = nullptr;
    uint8_t *base_ = nullptr;
    uint64_t size_ = 0;
    std::shared_ptr<AccOffloadMemManager> memMng_;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H
