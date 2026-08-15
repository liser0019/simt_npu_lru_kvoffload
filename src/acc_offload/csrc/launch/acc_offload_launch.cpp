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
#include <dlfcn.h>
#include "mf_file_util.h"
#include "acc_offload_launch.h"

namespace ock {
namespace offload {

bool AccOffloadLaunchApi::gLoaded = false;
std::mutex AccOffloadLaunchApi::gMutex;
void *AccOffloadLaunchApi::libHandle = nullptr;
const char *AccOffloadLaunchApi::gAccOffloadLibName = "libmf_hybm_accoffload.so";

AccOffloadSparseCopyFunc AccOffloadLaunchApi::pAccOffloadSparseCopy = nullptr;
AccOffloadLruCompactFunc AccOffloadLaunchApi::pAccOffloadLruCompact = nullptr;
AccOffloadComputeLruResidentAddrsFunc AccOffloadLaunchApi::pAccOffloadComputeLruResidentAddrs = nullptr;
AccOffloadSparseKvLoadRuntimeFunc AccOffloadLaunchApi::pAccOffloadSparseKvLoadRuntime = nullptr;
AccOffloadSparseKvPlanFsaRuntimeFunc
    AccOffloadLaunchApi::pAccOffloadSparseKvPlanFsaRuntime = nullptr;

int32_t AccOffloadLaunchApi::TryLoadLibrary()
{
    std::unique_lock<std::mutex> guard(gMutex);
    if (gLoaded) {
        return OFFLOAD_OK;
    }

    char *path = std::getenv("MEMFABRIC_HYBRID_EXTEND_LIB_PATH");
    if (path == nullptr) {
        OFFLOAD_LOG_WARN("Environment MEMFABRIC_HYBRID_EXTEND_LIB_PATH is not set.");
        return OFFLOAD_ERROR;
    }
    std::string libPath = std::string(path);
    if (!ock::mf::FileUtil::Realpath(libPath) || !ock::mf::FileUtil::IsDir(libPath)) {
        OFFLOAD_LOG_WARN("Environment MEMFABRIC_HYBRID_EXTEND_LIB_PATH check failed.");
        return OFFLOAD_ERROR;
    }

    std::string realPath;
    if (!ock::mf::FileUtil::LibraryRealPath(libPath, std::string(gAccOffloadLibName), realPath)) {
        OFFLOAD_LOG_WARN(libPath << " get lib path failed");
        return OFFLOAD_ERROR;
    }

    libHandle = dlopen(realPath.c_str(), RTLD_NOW | RTLD_NODELETE);
    if (libHandle == nullptr) {
        OFFLOAD_LOG_WARN("Failed to open library [" << realPath << "], error: " << dlerror());
        return OFFLOAD_FUNCTION_FAILED;
    }

    DL_LOAD_SYM_OPTIONAL(pAccOffloadSparseCopy, AccOffloadSparseCopyFunc, libHandle, "AccOffloadSparseCopy");
    DL_LOAD_SYM_OPTIONAL(pAccOffloadLruCompact, AccOffloadLruCompactFunc, libHandle, "AccOffloadLruCompact");
    DL_LOAD_SYM_OPTIONAL(pAccOffloadComputeLruResidentAddrs, AccOffloadComputeLruResidentAddrsFunc, libHandle,
                         "AccOffloadComputeLruResidentAddrs");
    DL_LOAD_SYM_OPTIONAL(pAccOffloadSparseKvLoadRuntime,
                         AccOffloadSparseKvLoadRuntimeFunc, libHandle,
                         "AccOffloadSparseKvLoadRuntime");
    DL_LOAD_SYM_OPTIONAL(pAccOffloadSparseKvPlanFsaRuntime,
                         AccOffloadSparseKvPlanFsaRuntimeFunc, libHandle,
                         "AccOffloadSparseKvPlanFsaRuntime");

    gLoaded = true;
    return OFFLOAD_OK;
}

void AccOffloadLaunchApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pAccOffloadSparseCopy = nullptr;
    pAccOffloadLruCompact = nullptr;
    pAccOffloadComputeLruResidentAddrs = nullptr;
    pAccOffloadSparseKvLoadRuntime = nullptr;
    pAccOffloadSparseKvPlanFsaRuntime = nullptr;

    if (libHandle != nullptr) {
        dlclose(libHandle);
        libHandle = nullptr;
    }

    gLoaded = false;
}

} // namespace offload
} // namespace ock
