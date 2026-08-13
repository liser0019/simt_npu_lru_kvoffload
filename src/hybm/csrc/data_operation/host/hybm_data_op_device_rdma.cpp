/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/
#include "hybm_data_op_device_rdma.h"

#include <sys/mman.h>
#include <cstdint>
#include <vector>

#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "hybm_def.h"
#include "hybm_define.h"
#include "hybm_logger.h"
#include "hybm_types.h"
#include "hybm_ptracer.h"
#include "hybm_gva.h"
#include "hybm_stream_manager.h"
#include "hybm_va_manager.h"
#include "mf_env_util.h"

namespace {
constexpr uint64_t RDMA_SWAP_SPACE_SIZE = 1024 * 1024 * 128;
}

namespace ock {
namespace mf {
static hybm_mem_type HybmDirectionSrcMemType[HYBM_DATA_COPY_DIRECTION_BUTT] = {
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_BUTT
};
static hybm_mem_type HybmDirectionDestMemType[HYBM_DATA_COPY_DIRECTION_BUTT] = {
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_HOST,
    HYBM_MEM_TYPE_DEVICE,
    HYBM_MEM_TYPE_BUTT
};

DataOpDeviceRDMA::DataOpDeviceRDMA(uint32_t rankId, std::shared_ptr<transport::TransportManager> tm) noexcept
    : rankId_{rankId}, transportManager_{std::move(tm)}
{}

Result DataOpDeviceRDMA::Initialize() noexcept
{
    if (inited_) {
        return BM_OK;
    }
    rdmaSwapSpaceSize_ = MfEnvUtil::GetOptionalUintOrDefault("HYBM_RDMA_SWAP_SPACE_SIZE", RDMA_SWAP_SPACE_SIZE);
    forceUnregistered_ =
        MfEnvUtil::GetOptionalUintOrDefault("HYBM_RDMA_FORCE_UNREGISTERED", 0) != 0;
    if (rdmaSwapSpaceSize_ == 0) {
        BM_LOG_INFO("HYBM_RDMA_SWAP_SPACE_SIZE is 0, skip swap memory allocation");
        inited_ = true;
        return BM_OK;
    }
    auto ret = AllocSwapMemory();
    if (ret != BM_OK) {
        return ret;
    }
    transport::TransportMemoryRegion input;
    input.addr = reinterpret_cast<uint64_t>(rdmaSwapBaseAddr_);
    input.size = rdmaSwapSpaceSize_;
    input.flags = transport::REG_MR_FLAG_ACL_DRAM;
    if (transportManager_ != nullptr) {
        ret = transportManager_->RegisterMemoryRegion(input);
        if (ret != BM_OK) {
            BM_LOG_ERROR("Failed to register rdma swap memory, size: " << rdmaSwapSpaceSize_);
            FreeSwapMemory();
            return BM_MALLOC_FAILED;
        }
    }
    rdmaSwapMemoryAllocator_ = std::make_shared<RbtreeRangePool>((uint8_t *)rdmaSwapBaseAddr_, rdmaSwapSpaceSize_);
    inited_ = true;
    return BM_OK;
}

void DataOpDeviceRDMA::UnInitialize() noexcept
{
    FreeSwapMemory();
    inited_ = false;
}

void DataOpDeviceRDMA::TransformVa(void *&src, void *&dst, hybm_data_copy_direction direction) noexcept
{
    // 对于本地内存, DRAM需要输入host va, HBM需要输入device va, transport才能识别
    // 对于远端内存, transport仅记录的gva, TransformVa输出应当为0
    uint64_t out;
    uint32_t oType = (HybmDirectionSrcMemType[direction] == HYBM_MEM_TYPE_HOST) ? HVM_HVA : HVM_DVA;
    out = HybmVaManager::GetInstance().TransformVa(reinterpret_cast<uint64_t>(src), HVM_GVA, oType);
    if (out != 0) {
        src = reinterpret_cast<void *>(out);
    }

    oType = (HybmDirectionDestMemType[direction] == HYBM_MEM_TYPE_HOST) ? HVM_HVA : HVM_DVA;
    out = HybmVaManager::GetInstance().TransformVa(reinterpret_cast<uint64_t>(dst), HVM_GVA, oType);
    if (out != 0) {
        dst = reinterpret_cast<void *>(out);
    }
}

Result DataOpDeviceRDMA::AllocSwapMemory()
{
    void *ptr = nullptr;
    int ret = DlHalApi::HalMemAlloc(&ptr, rdmaSwapSpaceSize_, MEM_HOST | MEM_TYPE_DDR | MEM_PAGE_HUGE);
    if (ret != 0) {
        BM_LOG_ERROR("Failed to HalMemAlloc rdma swap memory, size: " << rdmaSwapSpaceSize_ << " ret:" << ret);
        return BM_MALLOC_FAILED;
    }

    void *output;
    ret = DlHalApi::HalHostRegister(ptr, rdmaSwapSpaceSize_,
                                    HOST_MEM_MAP_DEV, HybmGetInitedLogicDeviceId(), &output);
    if (ret != 0) {
        BM_LOG_ERROR("Register swap mem failed, addr: " << ptr << " ret: " << ret);
        return ret;
    }
    ret = HybmVaManager::GetInstance().AddVaInfo({0, reinterpret_cast<uint64_t>(output),
        reinterpret_cast<uint64_t>(ptr), rdmaSwapSpaceSize_, HYBM_MEM_TYPE_HOST}, rankId_);
    if (ret != 0) {
        BM_LOG_ERROR("add va info failed, va:" << ptr << " ret:" << ret);
        FreeSwapMemory();
        return ret;
    }

    rdmaSwapBaseAddr_ = ptr;
    return BM_OK;
}

void DataOpDeviceRDMA::FreeSwapMemory()
{
    if (rdmaSwapBaseAddr_ != nullptr) {
        if (transportManager_ != nullptr) {
            const auto ret = transportManager_->UnregisterMemoryRegion((uint64_t)rdmaSwapBaseAddr_);
            if (ret != 0) {
                BM_LOG_ERROR("Failed to UnregisterMemoryRegion, ret: " << ret);
            }
        }
        DlHalApi::HalHostUnregisterEx(rdmaSwapBaseAddr_, HybmGetInitedLogicDeviceId(), HOST_MEM_MAP_DEV);
        const auto ret = DlHalApi::HalMemFree(rdmaSwapBaseAddr_);
        if (ret != 0) {
            BM_LOG_ERROR("Failed to HalMemFree swap memory, ret: " << ret);
        }
        HybmVaManager::GetInstance().RemoveOneVaInfo(reinterpret_cast<uint64_t>(rdmaSwapBaseAddr_), HVM_HVA);
        rdmaSwapBaseAddr_ = nullptr;
    }
}

DataOpDeviceRDMA::~DataOpDeviceRDMA()
{
    FreeSwapMemory();
    inited_ = false;
}

Result DataOpDeviceRDMA::DataCopy(hybm_copy_params &params, hybm_data_copy_direction direction,
                                  const ock::mf::ExtOptions &options) noexcept
{
    Result ret;
    TransformVa(params.src, params.dest, direction);
    switch (direction) {
        case HYBM_LOCAL_HOST_TO_GLOBAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_LH_TO_GH);
            ret = CopyLH2GH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_LH_TO_GH, ret);
            break;
        }
        case HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_LH_TO_GD);
            ret = CopyLH2GD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_LH_TO_GD, ret);
            break;
        }
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_LD_TO_GH);
            ret = CopyLD2GH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_LD_TO_GH, ret);
            break;
        }
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_LD_TO_GD);
            ret = CopyLD2GD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_LD_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GD_TO_GD);
            ret = CopyGD2GD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GD_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GD_TO_GH);
            ret = CopyGD2GH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GD_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GH_TO_GD);
            ret = CopyGH2GD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GH_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_GLOBAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GH_TO_GH);
            ret = CopyGH2GH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GH_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_LOCAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GH_TO_LH);
            ret = CopyGH2LH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GH_TO_LH, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GD_TO_LH);
            ret = CopyGD2LH(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GD_TO_LH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GH_TO_LD);
            ret = CopyGH2LD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GH_TO_LD, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE: {
            TP_TRACE_BEGIN(TP_HYBM_RDMA_GD_TO_LD);
            ret = CopyGD2LD(params.src, params.dest, params.dataSize, options);
            TP_TRACE_END(TP_HYBM_RDMA_GD_TO_LD, ret);
            break;
        }
        default:
            BM_LOG_ERROR("data copy invalid direction: " << direction);
            ret = BM_INVALID_PARAM;
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyLH2LH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    auto ret = DlAclApi::AclrtMemcpy(destVA, length, srcVA, length, ACL_MEMCPY_HOST_TO_HOST);
    if (ret != BM_OK) {
        BM_LOG_ERROR("AclrtMemcpy failed, ret: " << ret << " Src=" << VaToInfo(srcVA) <<
                     " dest=" << VaToInfo(destVA) << " length=" << length);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}
Result DataOpDeviceRDMA::CopyLD2LD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    auto ret = DlAclApi::AclrtMemcpy(destVA, length, srcVA, length, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != BM_OK) {
        BM_LOG_ERROR("AclrtMemcpy failed, ret: " << ret << " Src=" << VaToInfo(srcVA) <<
                     " dest=" << VaToInfo(destVA) << " length=" << length);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::CopyLH2LD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    auto ret = DlAclApi::AclrtMemcpy(destVA, length, srcVA, length, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != BM_OK) {
        BM_LOG_ERROR("AclrtMemcpy failed, ret: " << ret << " Src=" << VaToInfo(srcVA) <<
                     " dest=" << VaToInfo(destVA) << " length=" << length);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::CopyLD2LH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    auto ret = DlAclApi::AclrtMemcpy(destVA, length, srcVA, length, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != BM_OK) {
        BM_LOG_ERROR("AclrtMemcpy failed, ret: " << ret << " Src=" << VaToInfo(srcVA) <<
                     " dest=" << VaToInfo(destVA) << " length=" << length);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::CopyLH2GH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.destRankId == rankId_) {
        ret = CopyLH2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafePut(srcVA, destVA, length, options, true);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyLH2GD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.destRankId == rankId_) {
        ret = CopyLH2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafePut(srcVA, destVA, length, options, true);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyLD2GH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.destRankId == rankId_) {
        ret = CopyLD2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafePut(srcVA, destVA, length, options, false);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyLD2GD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.destRankId == rankId_) {
        ret = CopyLD2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafePut(srcVA, destVA, length, options, false);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyRDMA(const void *srcVA, void *destVA, uint64_t length,
                                  const ock::mf::ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    auto src = (uint64_t)(ptrdiff_t)srcVA;
    auto dest = (uint64_t)(ptrdiff_t)destVA;
    Result ret;
    if (options.srcRankId == rankId_) {
        ret = transportManager_->WriteRemote(options.destRankId, src, dest, length);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to write src to dest", ret);
    } else if (options.destRankId == rankId_) {
        ret = transportManager_->ReadRemote(options.srcRankId, dest, src, length);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to read src to dest", ret);
    } else {
        BM_LOG_ERROR("Invalid param, local rank:" << rankId_ << ", srcId: " << options.srcRankId
                                                  << ", dstId: " << options.destRankId);
        return BM_INVALID_PARAM;
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGH2GH(const void *srcVA, void *destVA, uint64_t length,
                                   const ock::mf::ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_ && options.destRankId == rankId_) {
        ret = CopyLH2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to rdma src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGD2GH(const void *srcVA, void *destVA, uint64_t length,
                                   const ock::mf::ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_ && options.destRankId == rankId_) {
        ret = CopyLD2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to rdma src to dest", ret);
    }
    return ret;
}
Result DataOpDeviceRDMA::CopyGH2GD(const void *srcVA, void *destVA, uint64_t length,
                                   const ock::mf::ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_ && options.destRankId == rankId_) {
        ret = CopyLH2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to rdma src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGD2GD(const void *srcVA, void *destVA, uint64_t length,
                                   const ock::mf::ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_ && options.destRankId == rankId_) {
        ret = CopyLD2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to rdma src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGH2LH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_) {
        ret = CopyLH2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafeGet(srcVA, destVA, length, options, true);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGD2LH(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_) {
        ret = CopyLD2LH(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafeGet(srcVA, destVA, length, options, true);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGH2LD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_) {
        ret = CopyLH2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafeGet(srcVA, destVA, length, options, false);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::CopyGD2LD(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options) noexcept
{
    BM_LOG_DEBUG("SrcVA=" << VaToInfo(srcVA) << ", destVA=" << VaToInfo(destVA) << ", length=" << length);
    Result ret;
    if (options.srcRankId == rankId_) {
        ret = CopyLD2LD(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    } else {
        ret = SafeGet(srcVA, destVA, length, options, false);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to dest", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::DataCopyAsync(hybm_copy_params &params, hybm_data_copy_direction direction,
                                       const ExtOptions &options) noexcept
{
    BM_LOG_ERROR("DataOpDeviceRDMA::DataCopyAsync Not Supported!");
    return BM_ERROR;
}

Result DataOpDeviceRDMA::Wait(int32_t waitId) noexcept
{
    // Since DataOpDeviceRDMA::DataCopyAsync is not supported, Wait should do nothing for now.
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchMergedWrite(hybm_batch_copy_params &swapParams, hybm_data_copy_direction direction,
                                          void **remote, const ExtOptions &options) noexcept
{
    int32_t ret;
    // Batch copy local data to swap memory
    if (HybmDirectionSrcMemType[direction] == HYBM_MEM_TYPE_HOST) {
        ret = BatchDataCopyLocalSync(swapParams, ACL_MEMCPY_HOST_TO_HOST, options);
    } else {
        ret = BatchDataCopyLocalBatch(swapParams, ACL_MEMCPY_DEVICE_TO_HOST, options);
    }
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to copy local data to swap memory: " << ret);
        return ret;
    }

    // Batch RDMA write from swap to remote, merge contiguous remote addresses
    Result errorCode = BM_OK;
    size_t segStart = 0;
    while (segStart < swapParams.batchSize) {
        size_t segEnd = segStart + 1;
        while (segEnd < swapParams.batchSize &&
               reinterpret_cast<uintptr_t>(remote[segEnd]) == reinterpret_cast<uintptr_t>(remote[segEnd - 1])
               + swapParams.dataSizes[segEnd - 1]) {
            ++segEnd;
        }
        uint64_t mergedSize = 0;
        for (size_t k = segStart; k < segEnd; ++k) {
            mergedSize += swapParams.dataSizes[k];
        }
        ret = transportManager_->WriteRemoteAsync(
                options.destRankId, reinterpret_cast<uint64_t>(swapParams.destinations[segStart]),
                reinterpret_cast<uint64_t>(remote[segStart]), mergedSize);
        if (ret != BM_OK) {
            errorCode = ret;
            BM_LOG_ERROR("Failed to write swap to remote ret: " << ret
                << " localRankId:" << rankId_ << " remoteRankId:" << options.destRankId);
            break;
        }
        segStart = segEnd;
    }
    ret = transportManager_->Synchronize(options.destRankId);
    if (ret != 0) {
        BM_LOG_ERROR("Failed to sync write remote ret: " << ret);
        return ret;
    }
    return errorCode;
}

Result DataOpDeviceRDMA::BatchMergedRead(hybm_batch_copy_params &swapParams, hybm_data_copy_direction direction,
                                          void **remote, const ExtOptions &options) noexcept
{
    Result errorCode = BM_OK;
    int32_t ret;
    size_t segStart = 0;
    while (segStart < swapParams.batchSize) {
        size_t segEnd = segStart + 1;
        while (segEnd < swapParams.batchSize &&
               reinterpret_cast<uintptr_t>(remote[segEnd]) == reinterpret_cast<uintptr_t>(remote[segEnd - 1]) +
                                                                      swapParams.dataSizes[segEnd - 1]) {
            ++segEnd;
        }
        uint64_t mergedSize = 0;
        for (size_t k = segStart; k < segEnd; ++k) {
            mergedSize += swapParams.dataSizes[k];
        }
        ret = transportManager_->ReadRemoteAsync(
                options.srcRankId, reinterpret_cast<uint64_t>(swapParams.sources[segStart]),
                reinterpret_cast<uint64_t>(remote[segStart]), mergedSize);
        if (ret != BM_OK) {
            errorCode = ret;
            BM_LOG_ERROR("Failed to read remote to swap ret: " << ret
                << " localRankId:" << rankId_ << " remoteRankId:" << options.srcRankId);
            break;
        }
        segStart = segEnd;
    }
    ret = transportManager_->Synchronize(options.srcRankId);
    if (ret != 0) {
        BM_LOG_ERROR("Failed to sync read remote ret: " << ret);
        return ret;
    }
    if (errorCode != BM_OK) {
        return errorCode;
    }

    if (HybmDirectionDestMemType[direction] == HYBM_MEM_TYPE_HOST) {
        ret = BatchDataCopyLocalSync(swapParams, ACL_MEMCPY_HOST_TO_HOST, options);
    } else {
        ret = BatchDataCopyLocalBatch(swapParams, ACL_MEMCPY_HOST_TO_DEVICE, options);
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchDataCopyDefault(hybm_batch_copy_params &params, hybm_data_copy_direction direction,
                                              const ExtOptions &options) noexcept
{
    Result ret = BM_OK;
    TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_DEFAULT);

    bool isWrite = (direction <= HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE);
    size_t batchSize = params.batchSize;

    auto tmpRdmaMemory = rdmaSwapMemoryAllocator_->Allocate(rdmaSwapSpaceSize_);
    void *tmpHost = tmpRdmaMemory.Address();
    if (tmpHost == nullptr) {
        BM_LOG_ERROR("Failed to malloc swap length: " << rdmaSwapSpaceSize_);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_DEFAULT, BM_MALLOC_FAILED);
        return BM_MALLOC_FAILED;
    }

    uint64_t batchOffset = 0;
    while (batchOffset < batchSize) {
        uint64_t currentBatchDataSize = 0;
        size_t batchEnd = batchOffset;
        while (batchEnd < batchSize && currentBatchDataSize + params.dataSizes[batchEnd] <= rdmaSwapSpaceSize_) {
            currentBatchDataSize += params.dataSizes[batchEnd];
            ++batchEnd;
        }

        if (currentBatchDataSize == 0) {
            BM_LOG_ERROR("Single count exceeds RDMA_SWAP_SPACE_SIZE: " << params.dataSizes[batchOffset]
                                                                       << " > " << rdmaSwapSpaceSize_);
            ret = BM_INVALID_PARAM;
            break;
        }

        size_t currentBatchSize = batchEnd - batchOffset;
        std::vector<void *> tmpSwapAddrs(currentBatchSize);
        std::vector<void *> tmpLocalAddrs(currentBatchSize);
        std::vector<uint64_t> tmpCounts(currentBatchSize);
        uint64_t offset = 0;
        for (size_t i = batchOffset; i < batchEnd; ++i) {
            tmpSwapAddrs[i - batchOffset] = static_cast<uint8_t *>(tmpHost) + offset;
            tmpLocalAddrs[i - batchOffset] = (isWrite ? params.sources[i] : params.destinations[i]);
            tmpCounts[i - batchOffset] = params.dataSizes[i];
            offset += params.dataSizes[i];
        }

        if (isWrite) {
            hybm_batch_copy_params swapParams = {tmpLocalAddrs.data(), tmpSwapAddrs.data(),
                                                  tmpCounts.data(), static_cast<uint32_t>(currentBatchSize)};
            TP_TRACE_BEGIN(TP_HYBM_RDMA_MERGE_WRITE);
            ret = BatchMergedWrite(swapParams, direction, &params.destinations[batchOffset], options);
            TP_TRACE_END(TP_HYBM_RDMA_MERGE_WRITE, ret);
        } else {
            hybm_batch_copy_params swapParams = {tmpSwapAddrs.data(), tmpLocalAddrs.data(),
                                                  tmpCounts.data(), static_cast<uint32_t>(currentBatchSize)};
            TP_TRACE_BEGIN(TP_HYBM_RDMA_MERGE_READ);
            ret = BatchMergedRead(swapParams, direction, &params.sources[batchOffset], options);
            TP_TRACE_END(TP_HYBM_RDMA_MERGE_READ, ret);
        }
        if (ret != BM_OK) {
            break;
        }
        batchOffset = batchEnd;
    }

    TP_TRACE_END(TP_HYBM_RDMA_BATCH_DEFAULT, ret);
    BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "[BatchDataCopy] Failed to copy src to dest", ret);
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchCopyLH2GD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyWrite(params, options, HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchCopyGD2LH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyRead(params, options, HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST);
}

Result DataOpDeviceRDMA::BatchDataCopyLocal(hybm_batch_copy_params &params, int32_t direction,
                                            const ock::mf::ExtOptions &options) noexcept
{
    switch (direction) {
        case HYBM_LOCAL_HOST_TO_GLOBAL_HOST:
        case HYBM_GLOBAL_HOST_TO_GLOBAL_HOST:
        case HYBM_GLOBAL_HOST_TO_LOCAL_HOST:
            return BatchDataCopyLocalSync(params, ACL_MEMCPY_HOST_TO_HOST, options);
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE:
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE:
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE:
            return BatchDataCopyLocalAsync(params, ACL_MEMCPY_DEVICE_TO_DEVICE, options);
        case HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE:
        case HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE:
            return BatchDataCopyLocalAsync(params, ACL_MEMCPY_HOST_TO_DEVICE, options);
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST:
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST:
            return BatchDataCopyLocalAsync(params, ACL_MEMCPY_DEVICE_TO_HOST, options);
        case HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE:
            return BatchDataCopyLocalBatch(params, ACL_MEMCPY_HOST_TO_DEVICE, options);
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST:
            return BatchDataCopyLocalBatch(params, ACL_MEMCPY_DEVICE_TO_HOST, options);
        default:
            BM_LOG_ERROR("Failed to BatchDataCopyLocal noy support direct:" << direction);
            return -1;
    }
}

Result DataOpDeviceRDMA::BatchDataCopyLocalSync(hybm_batch_copy_params &params, int32_t direction,
                                                const ExtOptions &options) noexcept
{
    for (size_t i = 0; i < params.batchSize; ++i) {
        auto destAddr = params.destinations[i];
        auto srcAddr = params.sources[i];
        auto count = params.dataSizes[i];
        auto ret = DlAclApi::AclrtMemcpy(destAddr, count, srcAddr, count, direction);
        if (ret != 0) {
            BM_LOG_ERROR("copy memory on local failed: " << ret << " direct:" << direction);
            return BM_DL_FUNCTION_FAILED;
        }
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchDataCopyLocalAsync(hybm_batch_copy_params &params, int32_t direction,
                                                 const ExtOptions &options) noexcept
{
    void *st = options.stream;
    auto ret = 0;
    uint32_t batchNum = params.batchSize;
    if (st == nullptr) {
        st = HybmStreamManager::GetThreadAclStream();
    }

    for (size_t i = 0; i < batchNum; ++i) {
        auto destAddr = params.destinations[i];
        auto srcAddr = params.sources[i];
        auto count = params.dataSizes[i];
        ret = DlAclApi::AclrtMemcpyAsync(destAddr, count, srcAddr, count, direction, st);
        if (ret != 0) {
            (void)DlAclApi::AclrtSynchronizeStream(st);
            BM_LOG_ERROR("copy memory on local failed: " << ret << " stream:" << reinterpret_cast<uintptr_t>(st)
                                                         << " direct:" << direction << std::hex << " src:" << srcAddr
                                                         << " dst:" << destAddr);
            return BM_DL_FUNCTION_FAILED;
        }
    }
    ret = DlAclApi::AclrtSynchronizeStream(st);
    if (ret != 0) {
        BM_LOG_ERROR("aclrtSynchronizeStream failed: " << ret << " stream:" << reinterpret_cast<uintptr_t>(st));
    }
    return ret;
}

Result DataOpDeviceRDMA::BatchDataCopyLocalBatch(hybm_batch_copy_params &params, int32_t direction,
                                                 const ExtOptions &options) noexcept
{
    uint32_t batchNum = params.batchSize;
    std::vector<aclrtMemcpyBatchAttr> attrs(batchNum);
    std::vector<size_t> attrsIds(batchNum);
    std::vector<size_t> sizes(batchNum);
    size_t idx = 0;
    auto deviceLoc = aclrtMemLocation{static_cast<uint32_t>(HybmGetInitDeviceId()),
                                      aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE};
    auto hostLoc = aclrtMemLocation{0, aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST};
    for (size_t i = 0; i < batchNum; i++) {
        if (direction == ACL_MEMCPY_HOST_TO_DEVICE) {
            attrs[i] = aclrtMemcpyBatchAttr{deviceLoc, hostLoc, {}};
        } else {
            attrs[i] = aclrtMemcpyBatchAttr{hostLoc, deviceLoc, {}};
        }
        attrsIds[i] = idx++;
        sizes[i] = params.dataSizes[i];
    }
    size_t fail_idx = 0;
    auto ret = DlAclApi::AclrtMemcpyBatch(params.destinations, sizes.data(), params.sources, sizes.data(), sizes.size(),
                                          attrs.data(), attrsIds.data(), attrs.size(), &fail_idx);
    if (ret != 0) {
        return BatchDataCopyLocalAsync(params, direction, options);
    }
    return ret;
}

void DataOpDeviceRDMA::ClassifyDataAddr(void **globalAddrs, void **localAddrs, const uint64_t *counts,
                                        uint32_t batchSize, std::unordered_map<uint32_t, CopyDescriptor> &registered,
                                        std::unordered_map<uint32_t, CopyDescriptor> &localed,
                                        std::unordered_map<uint32_t, CopyDescriptor> &notRegistered,
                                        uint32_t globalRankId) noexcept
{
    for (size_t i = 0; i < batchSize; ++i) {
        if (globalRankId == rankId_) {
            auto iter = localed.find(globalRankId);
            if (iter == localed.end()) {
                CopyDescriptor desc{};
                desc.localAddrs.push_back(localAddrs[i]);
                desc.globalAddrs.push_back(globalAddrs[i]);
                desc.counts.push_back(counts[i]);
                localed.emplace(std::make_pair(globalRankId, desc));
            } else {
                iter->second.localAddrs.push_back(localAddrs[i]);
                iter->second.globalAddrs.push_back(globalAddrs[i]);
                iter->second.counts.push_back(counts[i]);
            }
        } else if (forceUnregistered_ ||
                   !transportManager_->QueryHasRegistered((uint64_t)localAddrs[i], counts[i])) {
            auto iter = notRegistered.find(globalRankId);
            if (iter == notRegistered.end()) {
                CopyDescriptor desc{};
                desc.localAddrs.push_back(localAddrs[i]);
                desc.globalAddrs.push_back(globalAddrs[i]);
                desc.counts.push_back(counts[i]);
                notRegistered.emplace(std::make_pair(globalRankId, desc));
            } else {
                iter->second.localAddrs.push_back(localAddrs[i]);
                iter->second.globalAddrs.push_back(globalAddrs[i]);
                iter->second.counts.push_back(counts[i]);
            }
        } else {
            auto iter = registered.find(globalRankId);
            if (iter == registered.end()) {
                CopyDescriptor desc{};
                desc.localAddrs.push_back(localAddrs[i]);
                desc.globalAddrs.push_back(globalAddrs[i]);
                desc.counts.push_back(counts[i]);
                registered.emplace(std::make_pair(globalRankId, desc));
            } else {
                iter->second.localAddrs.push_back(localAddrs[i]);
                iter->second.globalAddrs.push_back(globalAddrs[i]);
                iter->second.counts.push_back(counts[i]);
            }
        }
    }
}

Result DataOpDeviceRDMA::BatchCopyWrite(hybm_batch_copy_params &params, const ExtOptions &options,
                                        hybm_data_copy_direction direction) noexcept
{
    auto ret = 0;
    ExtOptions tmpOptions = options;
    std::unordered_map<uint32_t, CopyDescriptor> localed{};
    std::unordered_map<uint32_t, CopyDescriptor> registered{};
    std::unordered_map<uint32_t, CopyDescriptor> notRegistered{};
    ClassifyDataAddr(params.destinations, params.sources, params.dataSizes, params.batchSize, registered, localed,
                     notRegistered, options.destRankId);

    // 先写异步
    std::set<uint32_t> asyncSubmittedRanks{};
    for (auto &it : registered) {
        hybm_batch_copy_params regParams = {it.second.localAddrs.data(), it.second.globalAddrs.data(),
                                            it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.destRankId = it.first;

        for (uint32_t i = 0; i < regParams.batchSize; ++i) {
            ret = transportManager_->WriteRemoteAsync(tmpOptions.destRankId, (uint64_t)regParams.sources[i],
                                                      (uint64_t)regParams.destinations[i], regParams.dataSizes[i]);
            if (ret != BM_OK) {
                for (uint32_t r : asyncSubmittedRanks) {
                    transportManager_->Synchronize(r);
                }
                BM_LOG_ERROR("Failed to write src to dest");
                return ret;
            }
            asyncSubmittedRanks.insert(it.first);
        }
    }
    // 再写本地
    for (auto &it : localed) {
        hybm_batch_copy_params localParams = {it.second.localAddrs.data(), it.second.globalAddrs.data(),
                                              it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.destRankId = it.first;
        TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LOCAL);
        ret = BatchDataCopyLocal(localParams, direction, tmpOptions);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_LOCAL, ret);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "write local failed:", ret);
    }
    // 再写未注册
    for (auto &it : notRegistered) {
        hybm_batch_copy_params notParams = {it.second.localAddrs.data(), it.second.globalAddrs.data(),
                                            it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.destRankId = it.first;
        ret = BatchDataCopyDefault(notParams, direction, tmpOptions);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "write default failed:", ret);
    }
    // 再等异步
    for (auto &it : registered) {
        TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_WAIT_W);
        ret = transportManager_->Synchronize(it.first);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_WAIT_W, ret);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to Synchronize", ret);
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchCopyRead(hybm_batch_copy_params &params, const ExtOptions &options,
                                       hybm_data_copy_direction direction) noexcept
{
    auto ret = 0;
    ExtOptions tmpOptions = options;
    std::unordered_map<uint32_t, CopyDescriptor> localed{};
    std::unordered_map<uint32_t, CopyDescriptor> registered{};
    std::unordered_map<uint32_t, CopyDescriptor> notRegistered{};
    ClassifyDataAddr(params.sources, params.destinations, params.dataSizes, params.batchSize, registered, localed,
                     notRegistered, options.srcRankId);

    // 先写异步
    std::set<uint32_t> asyncSubmittedRanks{};
    for (auto &it : registered) {
        hybm_batch_copy_params regParams = {it.second.globalAddrs.data(), it.second.localAddrs.data(),
                                            it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.srcRankId = it.first;
        for (uint32_t i = 0; i < regParams.batchSize; ++i) {
            ret = transportManager_->ReadRemoteAsync(tmpOptions.srcRankId, (uint64_t)regParams.destinations[i],
                                                     (uint64_t)regParams.sources[i], regParams.dataSizes[i]);
            if (ret != BM_OK) {
                for (uint32_t r : asyncSubmittedRanks) {
                    transportManager_->Synchronize(r);
                }
                BM_LOG_ERROR("Failed to read src to dest");
                return ret;
            }
            asyncSubmittedRanks.insert(it.first);
        }
    }
    // 再写本地
    for (auto &it : localed) {
        hybm_batch_copy_params localParams = {it.second.globalAddrs.data(), it.second.localAddrs.data(),
                                              it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.destRankId = it.first;
        TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LOCAL);
        ret = BatchDataCopyLocal(localParams, direction, tmpOptions);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_LOCAL, ret);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "read local failed:", ret);
    }
    // 再写未注册
    for (auto &it : notRegistered) {
        hybm_batch_copy_params notParams = {it.second.globalAddrs.data(), it.second.localAddrs.data(),
                                            it.second.counts.data(), static_cast<uint32_t>(it.second.counts.size())};
        tmpOptions.srcRankId = it.first;
        ret = BatchDataCopyDefault(notParams, direction, tmpOptions);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "write default failed:", ret);
    }
    // 再等异步
    for (auto &it : registered) {
        TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_WAIT_R);
        ret = transportManager_->Synchronize(it.first);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_WAIT_R, ret);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to Synchronize", ret);
    }
    return BM_OK;
}

Result DataOpDeviceRDMA::BatchCopyLD2GD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyWrite(params, options, HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchCopyLD2GH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyWrite(params, options, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST);
}

Result DataOpDeviceRDMA::BatchCopyGH2LD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyRead(params, options, HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchCopyGD2LD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyRead(params, options, HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchCopyLH2GH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyWrite(params, options, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
}

Result DataOpDeviceRDMA::BatchCopyG2G(hybm_batch_copy_params &params, const ExtOptions &options,
                                      hybm_data_copy_direction direction) noexcept
{
    auto ret = 0;
    auto batchSize = params.batchSize;
    std::set<uint32_t> asyncWriteRanks{};
    // 先写异步
    for (uint32_t i = 0; i < batchSize; i++) {
        auto srcRankId = options.srcRankId;
        auto dstRankId = options.destRankId;

        if (srcRankId == rankId_ && dstRankId == rankId_) {
            hybm_copy_params pm = {params.sources[i], params.destinations[i], params.dataSizes[i]};
            ret = DataCopy(pm, direction, options);
            BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "write default failed:", ret);
        } else if (srcRankId == rankId_) {
            ret = transportManager_->WriteRemoteAsync(options.destRankId, (uint64_t)params.sources[i],
                                                      (uint64_t)params.destinations[i], params.dataSizes[i]);
            if (ret != BM_OK) {
                for (uint32_t r : asyncWriteRanks) {
                    transportManager_->Synchronize(r);
                }
                BM_LOG_ERROR("Failed to write src to dest");
                return ret;
            }
            asyncWriteRanks.insert(options.destRankId);
        } else if (dstRankId == rankId_) {
            ret = transportManager_->ReadRemoteAsync(options.srcRankId, (uint64_t)params.destinations[i],
                                                     (uint64_t)params.sources[i], params.dataSizes[i]);
            if (ret != BM_OK) {
                for (uint32_t r : asyncWriteRanks) {
                    transportManager_->Synchronize(r);
                }
                BM_LOG_ERROR("Failed to read src to dest");
                return ret;
            }
            asyncWriteRanks.insert(options.srcRankId);
        } else {
            BM_LOG_ERROR("invalid param, local rank:" << rankId_ << ", srcId: " << srcRankId
                                                      << ", dstId: " << dstRankId);
            return BM_ERROR;
        }
    }
    // 再等异步
    for (auto &it : asyncWriteRanks) {
        TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_WAIT_W);
        ret = transportManager_->Synchronize(it);
        TP_TRACE_END(TP_HYBM_RDMA_BATCH_WAIT_W, ret);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to Synchronize", ret);
    }
    return ret;
}

Result DataOpDeviceRDMA::SafePut(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options,
                                 bool srcIsHost)
{
    Result ret = 0;
    uintptr_t srcBase = reinterpret_cast<uintptr_t>(srcVA);
    uintptr_t destBase = reinterpret_cast<uintptr_t>(destVA);
    uint64_t remainingLength = length;
    uint64_t offset = 0;
    if (transportManager_->QueryHasRegistered(srcBase, length)) {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy rdma", ret);
        return ret;
    }
    if (rdmaSwapSpaceSize_ == 0) {
        BM_LOG_ERROR("HYBM_RDMA_SWAP_SPACE_SIZE is 0, unable to copy unregistered addresses, srcVa: " << srcBase);
        return BM_ERROR;
    }
    while (remainingLength > 0) {
        uint64_t currentChunkSize = std::min(remainingLength, rdmaSwapSpaceSize_);
        auto tmpRdmaMemory = rdmaSwapMemoryAllocator_->Allocate(currentChunkSize);
        auto tmpHost = tmpRdmaMemory.Address();
        BM_ASSERT_LOG_AND_RETURN(tmpHost != nullptr, "Failed to malloc temp buffer", BM_MALLOC_FAILED);
        const void *currentSrc = reinterpret_cast<const void *>(srcBase + offset);
        void *currentDest = reinterpret_cast<void *>(destBase + offset);
        if (srcIsHost) {
            ret = CopyLH2LH(currentSrc, tmpHost, currentChunkSize, options);
        } else {
            ret = CopyLD2LH(currentSrc, tmpHost, currentChunkSize, options);
        }
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy src to tmp", ret);
        ret = CopyRDMA(tmpHost, currentDest, currentChunkSize, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy tmp to dest", ret);
        offset += currentChunkSize;
        remainingLength -= currentChunkSize;
    }
    return 0;
}

Result DataOpDeviceRDMA::SafeGet(const void *srcVA, void *destVA, uint64_t length, const ExtOptions &options,
                                 bool destIsHost)
{
    Result ret = 0;
    uintptr_t srcBase = reinterpret_cast<uintptr_t>(srcVA);
    uintptr_t destBase = reinterpret_cast<uintptr_t>(destVA);
    uint64_t remainingLength = length;
    uint64_t offset = 0;
    if (transportManager_->QueryHasRegistered(destBase, length)) {
        ret = CopyRDMA(srcVA, destVA, length, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "Failed to copy rdma", ret);
        return ret;
    }
    if (rdmaSwapSpaceSize_ == 0) {
        BM_LOG_ERROR("HYBM_RDMA_SWAP_SPACE_SIZE is 0, unable to copy unregistered addresses, srcVa: " << srcBase);
        return BM_ERROR;
    }
    while (remainingLength > 0) {
        uint64_t currentChunkSize = std::min(remainingLength, rdmaSwapSpaceSize_);
        auto tmpRdmaMemory = rdmaSwapMemoryAllocator_->Allocate(currentChunkSize);
        auto tmpHost = tmpRdmaMemory.Address();
        BM_ASSERT_LOG_AND_RETURN(tmpHost != nullptr, "[CopyGD2LH] Failed to malloc temp buffer", BM_MALLOC_FAILED);
        const void *currentSrc = reinterpret_cast<const void *>(srcBase + offset);
        void *currentDest = reinterpret_cast<void *>(destBase + offset);
        ret = CopyRDMA(currentSrc, tmpHost, currentChunkSize, options);
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "[CopyGD2LH] Failed to copy src to tmp", ret);
        if (destIsHost) {
            ret = CopyLH2LH(tmpHost, currentDest, currentChunkSize, options);
        } else {
            ret = CopyLH2LD(tmpHost, currentDest, currentChunkSize, options);
        }
        BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "[CopyGD2LH] Failed to copy tmp to dest", ret);
        offset += currentChunkSize;
        remainingLength -= currentChunkSize;
    }
    return 0;
}

Result DataOpDeviceRDMA::BatchCopyGH2GH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyG2G(params, options, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST);
}

Result DataOpDeviceRDMA::BatchCopyGH2GD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyG2G(params, options, HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchCopyGH2LH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyRead(params, options, HYBM_GLOBAL_HOST_TO_LOCAL_HOST);
}

Result DataOpDeviceRDMA::BatchCopyGD2GH(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyG2G(params, options, HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST);
}

Result DataOpDeviceRDMA::BatchCopyGD2GD(hybm_batch_copy_params &params, const ExtOptions &options) noexcept
{
    return BatchCopyG2G(params, options, HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE);
}

Result DataOpDeviceRDMA::BatchDataCopy(hybm_batch_copy_params &params, hybm_data_copy_direction direction,
                                       const ExtOptions &options) noexcept
{
    auto ret = 0;
    for (uint32_t i = 0; i < params.batchSize; i++) {
        TransformVa(params.sources[i], params.destinations[i], direction);
    }
    switch (direction) {
        case HYBM_LOCAL_HOST_TO_GLOBAL_HOST: { // 0
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LH_TO_GH);
            ret = BatchCopyLH2GH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_LH_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_GLOBAL_HOST: { // 4
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GH_TO_GH);
            ret = BatchCopyGH2GH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GH_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE: { // 5
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GH_TO_GD);
            ret = BatchCopyGH2GD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GH_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_LOCAL_HOST: { // 6
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GH_TO_LH);
            ret = BatchCopyGH2LH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GH_TO_LH, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST: { // 8
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GD_TO_GH);
            ret = BatchCopyGD2GH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GD_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE: { // 9
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GD_TO_GD);
            ret = BatchCopyGD2GD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GD_TO_GD, ret);
            break;
        }
        case HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE: { // 1
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LH_TO_GD);
            ret = BatchCopyLH2GD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_LH_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST: { // 10
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GD_TO_LH);
            ret = BatchCopyGD2LH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GD_TO_LH, ret);
            break;
        }
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST: { // 2
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LD_TO_GH);
            ret = BatchCopyLD2GH(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_LD_TO_GH, ret);
            break;
        }
        case HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE: { // 7
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GH_TO_LD);
            ret = BatchCopyGH2LD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GH_TO_LD, ret);
            break;
        }
        case HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE: { // 3
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_LD_TO_GD);
            ret = BatchCopyLD2GD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_LD_TO_GD, ret);
            break;
        }
        case HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE: { // 11
            TP_TRACE_BEGIN(TP_HYBM_RDMA_BATCH_GD_TO_LD);
            ret = BatchCopyGD2LD(params, options);
            TP_TRACE_END(TP_HYBM_RDMA_BATCH_GD_TO_LD, ret);
            break;
        }
        default: {
            ret = BM_ERROR;
            BM_LOG_ERROR("unexcepted direction:" << direction);
            break;
        }
    }
    return ret;
}
} // namespace mf
} // namespace ock
