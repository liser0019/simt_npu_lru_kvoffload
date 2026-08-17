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
#include "smem_net_common.h"
#include "smem_store_factory.h"
#include "acc_offload_launch.h"
#include "acc_offload_shared_dram_entry.h"
#include "acc_offload_store_port.h"

namespace ock {
namespace offload {

using namespace ock::smem;

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

int32_t AccOffloadSharedDramEntry::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (inited_) {
        return OFFLOAD_OK;
    }

    OFFLOAD_ASSERT_RETURN(config.worldSize != 0, OFFLOAD_ERROR);
    if (config.registerHostMemory != 0U && config.worldSize != 1U) {
        OFFLOAD_LOG_ERROR("registered Host pool currently supports only worldSize=1, worldSize: "
                          << config.worldSize);
        return OFFLOAD_ERROR;
    }
    int32_t ret = OFFLOAD_OK;
    do {
        ret = hybm_init(config.deviceId, 0);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("hybm_init failed, result: " << ret);
            break;
        }

        ret = AccOffloadLaunchApi::TryLoadLibrary();
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("offload launch load library failed");
            break;
        }

        uint16_t port = 0U;
        std::string portError;
        if (!internal::ResolveAccOffloadStorePort(config.deviceId, config.worldSize, port, portError)) {
            OFFLOAD_LOG_ERROR("invalid " << internal::ACC_OFFLOAD_PORT_BASE_ENV << ": " << portError);
            ret = OFFLOAD_ERROR;
            break;
        }
        std::string storeUrl = "tcp://127.0.0.1:" + std::to_string(port);
        storeUrl_ = storeUrl;
        OFFLOAD_LOG_INFO("using config store url: " << storeUrl);

        smem_bm_config_t bmCfg {};
        smem_bm_config_init(&bmCfg);
        bmCfg.rankId = config.rankId;
        bmCfg.autoRanking = false;
        bmCfg.startConfigStoreServer = config.rankId == 0;

        UrlExtraction extraction;
        if (extraction.ExtractIpPortFromUrl(storeUrl) != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("extract ip port from url failed, storeUrl: " << storeUrl);
            ret = OFFLOAD_ERROR;
            break;
        }

        StoreFactory::SetTlsInfo(bmCfg.storeTlsConfig);
        uint16_t model = bmCfg.startConfigStoreServer ? CSM_BOTH : CSM_CLIENT;
        auto confStore = StoreFactory::CreateStoreByUrl(storeUrl, model, config.worldSize, config.rankId);
        if (confStore == nullptr) {
            OFFLOAD_LOG_ERROR("create store failed, storeUrl: " << storeUrl);
            ret = OFFLOAD_ERROR;
            break;
        }

        auto prefix = "(" + std::to_string(HYBM_ENTITY_ID_OFFLOAD_BASE) + ")_";
        confStore = StoreFactory::PrefixStore(confStore, "OFFLOAD_");
        auto entryStore = StoreFactory::PrefixStore(confStore, prefix);

        SmemBmEntryOptions entryOpt {HYBM_ENTITY_ID_OFFLOAD_BASE - HYBM_ENTITY_ID_BM_BASE,
                                     config.rankId, bmCfg.dynamicWorldSize, bmCfg.controlOperationTimeout};
        bmEntry_ = SmMakeRef<SmemBmEntry>(entryOpt, entryStore);
        if (bmEntry_ == nullptr) {
            OFFLOAD_LOG_ERROR("create bm entry failed, rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        uint64_t alignedReserveSize = AlignUp(config.reserveSize, GB);
        uint64_t alignedAllocSize = AlignUp(config.allocSize, GB);
        smem_bm_create_option_t option {};
        option.maxDramSize = alignedReserveSize;
        option.localDRAMSize = alignedAllocSize;
        option.maxHbmSize = 0;
        option.localHBMSize = 0;
        option.dataOpType = SMEMB_DATA_OP_MTE;
        option.flags = SMEM_BM_FLAG_DRAM_MAP_HOST_VA;
        registerHostMemory_ = config.registerHostMemory != 0U;
        if (registerHostMemory_) {
            option.flags |= SMEM_BM_FLAG_HOST_REGISTER_FOR_DEVICE;
        }
        option.dramShmFd = -1;
        option.enable56BitsGva = false;

        ret = SmemBmEntryInitWithOptions(bmEntry_, &option, config.rankId, config.deviceId, config.worldSize,
                                         storeUrl, bmCfg.hcomTlsConfig);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("SmemBmEntryInitWithOptions failed, rankId: " << config.rankId
                              << ", reserveSize: " << alignedReserveSize << ", allocSize: " << alignedAllocSize);
            ret = OFFLOAD_ERROR;
            break;
        }

        ret = bmEntry_->Join(0);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("bm entry join failed, result: " << ret << ", rankId: " << config.rankId);
            break;
        }

        void *gva = bmEntry_->GetHostGvaAddress();
        if (gva == nullptr) {
            OFFLOAD_LOG_ERROR("get host gva failed");
            ret = OFFLOAD_ERROR;
            break;
        }
        base_ = reinterpret_cast<uint8_t *>(gva) + bmEntry_->GetCoreOptions().maxDRAMSize * config.rankId;
        size_ = alignedReserveSize;

        memMng_ = std::make_shared<AccOffloadMemManager>(base_, size_);
        if (memMng_ == nullptr) {
            OFFLOAD_LOG_ERROR("create mem manager failed");
            ret = OFFLOAD_ERROR;
            break;
        }
    } while (0);

    inited_ = true;
    if (ret != OFFLOAD_OK) {
        UnInitalize();
        return ret;
    }

    OFFLOAD_LOG_INFO("shared dram entry initialized, rankId: " << config.rankId << ", deviceId: " << config.deviceId
                     << ", base: " << reinterpret_cast<void *>(base_) << ", size: " << size_);
    return OFFLOAD_OK;
}

void AccOffloadSharedDramEntry::UnInitalize()
{
    if (!inited_) {
        return;
    }

    memMng_.reset();
    if (bmEntry_ != nullptr) {
        bmEntry_->UnInitalize();
        bmEntry_ = nullptr;
    }
    if (!storeUrl_.empty()) {
        StoreFactory::DestroyStore(storeUrl_);
        storeUrl_.clear();
    }
    AccOffloadLaunchApi::CleanupLibrary();
    hybm_uninit();

    base_ = nullptr;
    size_ = 0;
    registerHostMemory_ = false;
    inited_ = false;
}

void *AccOffloadSharedDramEntry::MallocHost(size_t size)
{
    if (memMng_ == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr, malloc failed");
        return nullptr;
    }

    OFFLOAD_LOG_DEBUG("shared malloc host size: " << size);
    return memMng_->Allocate(size);
}

void AccOffloadSharedDramEntry::FreeHost(void *ptr)
{
    if (memMng_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr or ptr is nullptr, free failed");
        return;
    }

    OFFLOAD_LOG_DEBUG("shared free host ptr: " << reinterpret_cast<uint64_t>(ptr));
    memMng_->Release(ptr);
}

uint64_t AccOffloadSharedDramEntry::GetDeviceAddress(const void *ptr, size_t size)
{
    if (!registerHostMemory_ || !RangeInPool(base_, size_, ptr, size)) {
        OFFLOAD_LOG_ERROR("host address is not in a device-registered shared pool, ptr: "
                          << reinterpret_cast<uint64_t>(ptr) << ", size: " << size);
        return 0U;
    }

    uint64_t deviceAddress = 0U;
    auto ret = hybm_gva_to_va(reinterpret_cast<uint64_t>(ptr), HYBM_MEM_TYPE_DEVICE, &deviceAddress);
    if (ret != OFFLOAD_OK || deviceAddress == 0U) {
        OFFLOAD_LOG_ERROR("convert shared host GVA to DVA failed, ptr: " << reinterpret_cast<uint64_t>(ptr)
                          << ", size: " << size << ", ret: " << ret);
        return 0U;
    }
    return deviceAddress;
}

int32_t AccOffloadSharedDramEntry::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                               uint32_t *sizePtr, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("shared sparse copy, src: " << reinterpret_cast<uint64_t>(srcPtrs) <<
                      ", dst: " << reinterpret_cast<uint64_t>(dstPtrs) <<
                      ", len: " << reinterpret_cast<uint64_t>(lenPtrs) <<
                      ", size: " << *sizePtr << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx);
}

int32_t AccOffloadSharedDramEntry::LruResidentCompact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                                       uint64_t stable_prefix_lens, uint64_t slot_to_token,
                                                       uint64_t lru_slots, uint64_t current_slots,
                                                       uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                                       uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                                       uint64_t epochs, int64_t num_reqs, int64_t topk,
                                                       int64_t capacity, int64_t max_token, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("shared lru resident compact, num_reqs: " << num_reqs << ", topk: " << topk
                      << ", capacity: " << capacity << ", max_token: " << max_token << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadLruCompact(req_ids, last_req_ids, topk_indices, stable_prefix_lens,
                                                      slot_to_token, lru_slots, current_slots, miss_count,
                                                      miss_tokens, miss_slots, token_mark_workspace,
                                                      token_pos_workspace, epochs, num_reqs, topk, capacity,
                                                      max_token, devIdx);
}

int32_t AccOffloadSharedDramEntry::ComputeLruResidentAddrs(uint64_t miss_count, uint64_t miss_tokens,
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
    OFFLOAD_LOG_DEBUG("shared compute lru resident addrs, num_reqs: " << num_reqs << ", topk: " << topk
                      << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadComputeLruResidentAddrs(
        miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer, size_buffer, num_tokens_buffer,
        block_size, token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks, devIdx);
}

int32_t AccOffloadSharedDramEntry::SparseKvLoadRuntime(
    const sparse_kv_load_runtime_params_t &params, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("shared sparse kv load runtime, num_reqs: "
                      << params.num_reqs << ", topk: " << params.topk
                      << ", capacity: " << params.capacity
                      << ", max_token: " << params.max_token
                      << ", devIdx: " << devIdx);
    return AccOffloadLaunchApi::AccOffloadSparseKvLoadRuntime(params, devIdx);
}

} // namespace offload
} // namespace ock
