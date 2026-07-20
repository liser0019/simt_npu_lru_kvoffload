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
#include "acc_offload_entry_manager.h"
#include "acc_offload_local_dram_entry.h"
#include "acc_offload_shared_dram_entry.h"
#include "acc_offload_define.h"

namespace ock {
namespace offload {

AccOffloadEntryManager &AccOffloadEntryManager::Instance()
{
    static AccOffloadEntryManager instance;
    return instance;
}

AccOffloadEntryManager::~AccOffloadEntryManager()
{
    UnInitalize();
}

int32_t AccOffloadEntryManager::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (inited_) {
        OFFLOAD_LOG_WARN("entry manager already initialized, scene: " << static_cast<uint32_t>(scene_));
        return OFFLOAD_OK;
    }

    switch (config.scene) {
        case OFFLOAD_SCENE_LOCAL:
            entry_ = std::make_unique<AccOffloadLocalDramEntry>();
            break;
        case OFFLOAD_SCENE_SHARED:
            entry_ = std::make_unique<AccOffloadSharedDramEntry>();
            break;
        default:
            OFFLOAD_LOG_ERROR("invalid scene: " << static_cast<uint32_t>(config.scene));
            return OFFLOAD_ERROR;
    }
    scene_ = config.scene;

    auto ret = entry_->Initialize(config);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("entry initialize failed, scene: " << static_cast<uint32_t>(scene_)
                          << ", ret: " << ret);
        entry_->UnInitalize();
        entry_.reset();
        scene_ = OFFLOAD_SCENE_LOCAL;
        return ret;
    }

    inited_ = true;
    OFFLOAD_LOG_INFO("entry manager initialized, scene: " << static_cast<uint32_t>(scene_)
                      << ", deviceId: " << config.deviceId << ", reserveSize: " << config.reserveSize
                      << ", allocSize: " << config.allocSize
                      << ", worldSize: " << config.worldSize << ", rankId: " << config.rankId);
    return OFFLOAD_OK;
}

void AccOffloadEntryManager::UnInitalize()
{
    if (!inited_) {
        return;
    }

    if (entry_ != nullptr) {
        entry_->UnInitalize();
    }
    entry_.reset();
    inited_ = false;
    OFFLOAD_LOG_INFO("entry manager uninitialized, scene: " << static_cast<uint32_t>(scene_));
}

void *AccOffloadEntryManager::MallocHost(size_t size)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, malloc failed, size: " << size);
        return nullptr;
    }
    return entry_->MallocHost(size);
}

void AccOffloadEntryManager::FreeHost(void *ptr)
{
    if (entry_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null or ptr is null, free failed");
        return;
    }
    entry_->FreeHost(ptr);
}

int32_t AccOffloadEntryManager::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                           uint32_t *sizePtr, uint8_t devIdx)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, sparse copy failed");
        return OFFLOAD_ERROR;
    }
    return entry_->SparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx);
}

int32_t AccOffloadEntryManager::LruResidentCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                                    uint64_t stable_prefix_lens, uint64_t slot_to_token,
                                                    uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
                                                    uint64_t miss_tokens, uint64_t miss_slots,
                                                    uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                                    uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity,
                                                    int64_t max_token, uint8_t devIdx)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, lru resident compact failed");
        return OFFLOAD_ERROR;
    }
    return entry_->LruResidentCompact(req_ids, last_req_ids, topk_indices, stable_prefix_lens, slot_to_token, lru_slots,
                                       current_slots, miss_count, miss_tokens, miss_slots, token_mark_workspace,
                                       token_pos_workspace, epochs, num_reqs, topk, capacity, max_token, devIdx);
}

int32_t AccOffloadEntryManager::ComputeLruResidentAddrs(uint64_t miss_count, uint64_t miss_tokens,
                                                         uint64_t miss_slots, uint64_t block_table,
                                                         uint64_t gvas_buffer, uint64_t addr_buffer,
                                                         uint64_t size_buffer, uint64_t num_tokens_buffer,
                                                         int32_t block_size, int32_t token_size_bytes_k,
                                                         int32_t token_size_bytes_v, int64_t gvas_k_base,
                                                         int64_t gvas_v_base, int64_t addr_k_base,
                                                         int64_t addr_v_base, int32_t resident_capacity,
                                                         int64_t num_reqs, int64_t topk, int64_t max_num_blocks,
                                                         uint8_t devIdx)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, compute lru resident addrs failed");
        return OFFLOAD_ERROR;
    }
    return entry_->ComputeLruResidentAddrs(miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer,
                                           size_buffer, num_tokens_buffer, block_size, token_size_bytes_k,
                                           token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
                                           resident_capacity, num_reqs, topk, max_num_blocks, devIdx);
}

} // namespace offload
} // namespace ock
