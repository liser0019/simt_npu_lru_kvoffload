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

int32_t AccOffloadSharedDramEntry::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (inited_) {
        return OFFLOAD_OK;
    }

    OFFLOAD_ASSERT_RETURN(config.worldSize != 0, OFFLOAD_ERROR);
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

        constexpr int portBase = 8500;
        int port = portBase + config.deviceId / config.worldSize;
        std::string storeUrl = "tcp://127.0.0.1:" + std::to_string(port);
        storeUrl_ = storeUrl;

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

int32_t AccOffloadSharedDramEntry::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                              uint32_t *sizePtr, uint8_t devIdx)
{
    OFFLOAD_LOG_DEBUG("shared sparse copy, src: " << reinterpret_cast<uint64_t>(srcPtrs) <<
                      ", dst: " << reinterpret_cast<uint64_t>(dstPtrs) <<
                      ", len: " << reinterpret_cast<uint64_t>(lenPtrs) <<
                      ", size: " << *sizePtr << ", devIdx: " << devIdx);

    return AccOffloadLaunchApi::AccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx);
}

} // namespace offload
} // namespace ock
