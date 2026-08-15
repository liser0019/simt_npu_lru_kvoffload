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
#include "hybm_big_mem.h"
#include "acc_offload_launch.h"
#include "acc_offload_local_dram_entry.h"

namespace ock {
namespace offload {

using namespace ock::mf;

constexpr uint64_t KB = 1024ULL;
constexpr uint64_t MB = KB * 1024ULL;
constexpr uint64_t GB = MB * 1024ULL;

static uint64_t AlignUp(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1) & ~(align - 1);
}

static bool RangeInPool(const uint8_t *base, uint64_t poolSize, const void *ptr, size_t size) noexcept
{
    if (base == nullptr || ptr == nullptr || size == 0U) {
        return false;
    }
    auto baseAddress = reinterpret_cast<uint64_t>(base);
    auto address = reinterpret_cast<uint64_t>(ptr);
    if (address < baseAddress) {
        return false;
    }
    auto offset = address - baseAddress;
    return offset <= poolSize && size <= poolSize - offset;
}

int32_t AccOffloadLocalDramEntry::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (inited_) {
        return OFFLOAD_OK;
    }

    uint32_t flags = 0;
    int32_t ret = hybm_init(config.deviceId, flags);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("init hybm failed, result: " << ret);
        hybm_uninit();
        return OFFLOAD_ERROR;
    }

    uint64_t alignedReserveSize = AlignUp(config.reserveSize, GB);
    uint64_t alignedAllocSize = AlignUp(config.allocSize, GB);
    if (alignedReserveSize != alignedAllocSize) {
        OFFLOAD_LOG_ERROR("local dram requires reserveSize == allocSize, reserveSize: " << config.reserveSize
                          << ", allocSize: " << config.allocSize);
        hybm_uninit();
        return OFFLOAD_ERROR;
    }
    hybm_options options{};
    options.bmType = HYBM_TYPE_HOST_INITIATE;
    options.memType = HYBM_MEM_TYPE_HOST;
    options.bmDataOpType = HYBM_DOP_TYPE_MTE;
    options.rankCount = 1;
    options.rankId = 0;
    options.devId = config.deviceId;
    options.maxDRAMSize = alignedReserveSize;
    options.hostVASpace = alignedAllocSize;
    options.scene = HYBM_SCENE_DEFAULT;
    options.flags = HYBM_FLAG_DRAM_MAP_HOST_VA;
    registerHostMemory_ = config.registerHostMemory != 0U;
    if (registerHostMemory_) {
        options.flags |= HYBM_FLAG_HOST_REGISTER_FOR_DEVICE;
    }
    options.dramShmFd = -1;

    do {
        ret = AccOffloadLaunchApi::TryLoadLibrary();
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("offload launch load library failed");
            ret = OFFLOAD_ERROR;
            break;
        }

        entity_ = hybm_create_entity(HYBM_ENTITY_ID_OFFLOAD_BASE, &options, flags);
        if (entity_ == nullptr) {
            OFFLOAD_LOG_ERROR("create entity failed");
            ret = OFFLOAD_ERROR;
            break;
        }

        ret = hybm_reserve_mem_space(entity_, flags);
        if (ret != 0) {
            OFFLOAD_LOG_ERROR("reserve mem failed, result: " << ret);
            ret = OFFLOAD_ERROR;
            break;
        }

        if (options.maxDRAMSize > 0) {
            slice_ = hybm_alloc_local_memory(entity_, HYBM_MEM_TYPE_HOST, options.hostVASpace, flags);
            if (slice_ == nullptr) {
                OFFLOAD_LOG_ERROR("alloc local host mem failed, size: " << options.hostVASpace);
                ret = OFFLOAD_ERROR;
                break;
            }
        }

        base_ = reinterpret_cast<uint8_t *>(hybm_get_slice_va(entity_, slice_));
        if (base_ == nullptr) {
            OFFLOAD_LOG_ERROR("get slice va failed");
            ret = OFFLOAD_ERROR;
            break;
        }
    } while (0);

    size_ = alignedReserveSize;
    memMng_ = std::make_shared<AccOffloadMemManager>(base_, size_);
    inited_ = true;

    if (ret != 0 || memMng_ == nullptr) {
        UnInitalize();
        return ret;
    }

    return ret;
}

void AccOffloadLocalDramEntry::UnInitalize()
{
    if (!inited_) {
        return;
    }

    uint32_t flags = 0;
    if (entity_ != nullptr) {
        if (slice_ != nullptr) {
            hybm_free_local_memory(entity_, slice_, 1, flags);
        }
        hybm_unreserve_mem_space(entity_, flags);
        hybm_destroy_entity(entity_, flags);
    }

    AccOffloadLaunchApi::CleanupLibrary();

    hybm_uninit();
    entity_ = nullptr;
    registerHostMemory_ = false;
    inited_ = false;
}

void *AccOffloadLocalDramEntry::MallocHost(size_t size)
{
    if (memMng_ == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr, malloc failed");
        return nullptr;
    }

    OFFLOAD_LOG_DEBUG("malloc host size: " << size);
    return memMng_->Allocate(size);
}

void AccOffloadLocalDramEntry::FreeHost(void *ptr)
{
    if (memMng_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr or ptr is nullptr, free failed");
        return;
    }

    OFFLOAD_LOG_DEBUG("free host ptr: " << reinterpret_cast<uint64_t>(ptr));
    memMng_->Release(ptr);
}

uint64_t AccOffloadLocalDramEntry::GetDeviceAddress(const void *ptr, size_t size)
{
    if (!registerHostMemory_ || !RangeInPool(base_, size_, ptr, size)) {
        OFFLOAD_LOG_ERROR("host address is not in a device-registered local pool, ptr: "
                          << reinterpret_cast<uint64_t>(ptr) << ", size: " << size);
        return 0U;
    }

    uint64_t deviceAddress = 0U;
    auto ret = hybm_gva_to_va(reinterpret_cast<uint64_t>(ptr), HYBM_MEM_TYPE_DEVICE, &deviceAddress);
    if (ret != OFFLOAD_OK || deviceAddress == 0U) {
        OFFLOAD_LOG_ERROR("convert local host GVA to DVA failed, ptr: " << reinterpret_cast<uint64_t>(ptr)
                          << ", size: " << size << ", ret: " << ret);
        return 0U;
    }
    return deviceAddress;
}

int32_t AccOffloadLocalDramEntry::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                             uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("sparse copy, src: " << reinterpret_cast<uint64_t>(srcPtrs) <<
                      ", dst: " << reinterpret_cast<uint64_t>(srcPtrs) <<
                      ", len: " << reinterpret_cast<uint64_t>(lenPtrs) <<
                      ", size: " << *sizePtr << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx);
}

int32_t AccOffloadLocalDramEntry::LruResidentCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                                      uint64_t stable_prefix_lens, uint64_t slot_to_token,
                                                      uint64_t lru_slots, uint64_t current_slots,
                                                      uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                                      uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                                      uint64_t epochs, int64_t num_reqs, int64_t topk,
                                                      int64_t capacity, int64_t max_token, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("lru resident compact, num_reqs: " << num_reqs << ", topk: " << topk
                      << ", capacity: " << capacity << ", max_token: " << max_token << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadLruCompact(req_ids, last_req_ids, topk_indices, stable_prefix_lens,
                                                      slot_to_token, lru_slots, current_slots, miss_count,
                                                      miss_tokens, miss_slots, token_mark_workspace,
                                                      token_pos_workspace, epochs, num_reqs, topk, capacity,
                                                      max_token, devIdx);
}

int32_t AccOffloadLocalDramEntry::ComputeLruResidentAddrs(uint64_t miss_count, uint64_t miss_tokens,
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
    OFFLOAD_LOG_DEBUG("compute lru resident addrs, num_reqs: " << num_reqs << ", topk: " << topk
                      << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadComputeLruResidentAddrs(
        miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer, size_buffer, num_tokens_buffer,
        block_size, token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks, devIdx);
}

int32_t AccOffloadLocalDramEntry::SparseKvLoadRuntime(
    const sparse_kv_load_runtime_params_t &params, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("sparse kv load runtime, num_reqs: " << params.num_reqs
                      << ", topk: " << params.topk
                      << ", capacity: " << params.capacity
                      << ", max_token: " << params.max_token
                      << ", devIdx: " << devIdx);
    return AccOffloadLaunchApi::AccOffloadSparseKvLoadRuntime(params, devIdx);
}

int32_t AccOffloadLocalDramEntry::SparseKvPlanFsaRuntime(
    const sparse_kv_plan_fsa_runtime_params_t &params, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("sparse kv FSA plan runtime, logical rows: "
                      << params.num_logical_rows << ", physical rows: "
                      << params.physical_row_capacity << ", topk: "
                      << params.topk << ", capacity: " << params.capacity
                      << ", devIdx: " << devIdx);
    return AccOffloadLaunchApi::AccOffloadSparseKvPlanFsaRuntime(
        params, devIdx);
}

} // namespace offload
} // namespace ock
