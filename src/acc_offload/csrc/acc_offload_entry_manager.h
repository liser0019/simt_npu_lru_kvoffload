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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_MANAGER_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_MANAGER_H

#include <mutex>
#include <memory>
#include "acc_offload.h"
#include "acc_offload_entry.h"

namespace ock {
namespace offload {

/*
 * Singleton manager that owns the single active memory-pool entry.
 *
 * The concrete entry (AccOffloadLocalDramEntry or AccOffloadSharedDramEntry) is
 * selected from offload_config_t::scene, so the dispatch policy is driven by the
 * application layer rather than inferred from worldSize. The C entry point
 * (acc_offload.cpp) talks only to this manager, which keeps the software layered:
 *
 *   python -> acc_offload.cpp (C API) -> AccOffloadEntryManager -> AccOffloadEntry
 */
class AccOffloadEntryManager {
public:
    static AccOffloadEntryManager &Instance();

    AccOffloadEntryManager() = default;
    ~AccOffloadEntryManager();

    AccOffloadEntryManager(const AccOffloadEntryManager &) = delete;
    AccOffloadEntryManager &operator=(const AccOffloadEntryManager &) = delete;

    int32_t Initialize(const offload_config_t &config);

    void UnInitalize();

    void *MallocHost(size_t size);

    void FreeHost(void *ptr);

    int32_t SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, uint8_t devIdx);

    inline bool IsInitialized() const
    {
        return inited_;
    }

    inline offload_scene_t GetScene() const
    {
        return scene_;
    }

private:
    std::mutex mutex_;
    bool inited_ = false;
    offload_scene_t scene_ = OFFLOAD_SCENE_LOCAL;
    std::unique_ptr<AccOffloadEntry> entry_;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_MANAGER_H
