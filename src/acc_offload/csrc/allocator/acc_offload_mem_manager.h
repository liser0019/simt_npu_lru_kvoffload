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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_MEM_MANAGER_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_MEM_MANAGER_H

#include <map>
#include <set>
#include "acc_offload_define.h"

namespace ock {
namespace offload {

struct MemoryRange {
    const uint64_t offset;
    const uint64_t size;

    MemoryRange(uint64_t o, uint64_t s) noexcept : offset{o}, size{s} {}
};

struct RangeSizeFirstComparator {
    bool operator()(const MemoryRange &mr1, const MemoryRange &mr2) const noexcept
    {
        if (mr1.size != mr2.size) {
            return mr1.size < mr2.size;
        }

        return mr1.offset < mr2.offset;
    }
};

class AccOffloadMemManager {
public:
    AccOffloadMemManager(void *base, uint64_t size) noexcept : base_{reinterpret_cast<uint8_t *>(base)}, size_{size}
    {
        addressIdleTree_[0] = size;
        sizeIdleTree_.insert({0, size});
    }

    ~AccOffloadMemManager() = default;
    AccOffloadMemManager(const AccOffloadMemManager &) = delete;
    AccOffloadMemManager &operator=(const AccOffloadMemManager &) = delete;

public:
    void *Allocate(uint64_t size) noexcept
    {
        if (size == 0 || size > size_ || base_ == nullptr) {
            OFFLOAD_LOG_ERROR("memory manager allocate fail, size:" << size << ", size_:" << size_);
            return nullptr;
        }

        auto aligned_size = AllocatedSizeAlignUp(size);
        MemoryRange anchor{0, aligned_size};

        auto size_pos = sizeIdleTree_.lower_bound(anchor);
        if (size_pos == sizeIdleTree_.end()) {
            OFFLOAD_LOG_ERROR("memory manager allocate fail, offset:" << anchor.offset << ", size:" << anchor.size);
            return nullptr;
        }

        auto targetOffset = size_pos->offset;
        auto targetSize = size_pos->size;
        auto addr_pos = addressIdleTree_.find(targetOffset);
        if (addr_pos == addressIdleTree_.end()) {
            OFFLOAD_LOG_ERROR("memory manager allocate fail, targetOffset:" << targetOffset);
            return nullptr;
        }

        sizeIdleTree_.erase(size_pos);
        addressIdleTree_.erase(addr_pos);
        addressUsedTree_.emplace(targetOffset, aligned_size);
        if (targetSize > aligned_size) {
            MemoryRange left{targetOffset + aligned_size, targetSize - aligned_size};
            addressIdleTree_.emplace(left.offset, left.size);
            sizeIdleTree_.emplace(left);
        }

        return base_ + targetOffset;
    }

    int32_t Release(void *address) noexcept
    {
        auto u8a = reinterpret_cast<uint8_t *>(address);
        if (u8a < base_ || u8a >= base_ + size_) {
            OFFLOAD_LOG_ERROR("memory manager release fail, not in range");
            return -1;
        }

        auto offset = u8a - base_;
        auto pos = addressUsedTree_.find(offset);
        if (pos == addressUsedTree_.end()) {
            OFFLOAD_LOG_ERROR("memory manager release fail, offset:" << offset);
            return -1;
        }

        auto size = pos->second;
        uint64_t finalOffset = static_cast<uint64_t>(offset);
        uint64_t finalSize = size;
        addressUsedTree_.erase(pos);

        auto prevAddrPos = addressIdleTree_.lower_bound(offset);
        if (prevAddrPos != addressIdleTree_.begin()) {
            --prevAddrPos;
            if (prevAddrPos != addressIdleTree_.end() &&
                prevAddrPos->first + prevAddrPos->second == static_cast<uint64_t>(offset)) {
                finalOffset = prevAddrPos->first;
                finalSize += prevAddrPos->second;

                auto prevAddrRange = *prevAddrPos;
                addressIdleTree_.erase(prevAddrPos);
                sizeIdleTree_.erase(MemoryRange{prevAddrRange.first, prevAddrRange.second});
            }
        }

        auto nextAddrPos = addressIdleTree_.find(offset + size);
        if (nextAddrPos != addressIdleTree_.end()) {
            uint64_t nextAddr = nextAddrPos->first;
            uint64_t nextSize = nextAddrPos->second;
            finalSize += nextSize;
            addressIdleTree_.erase(nextAddrPos);
            sizeIdleTree_.erase(MemoryRange{nextAddr, nextSize});
        }
        addressIdleTree_.emplace(finalOffset, finalSize);
        sizeIdleTree_.emplace(MemoryRange{finalOffset, finalSize});
        return 0;
    }

private:
    static uint64_t AllocatedSizeAlignUp(uint64_t inputSize) noexcept
    {
        constexpr uint64_t alignSize = 16UL;
        constexpr uint64_t alignSizeMask = ~(alignSize - 1UL);
        return (inputSize + alignSize - 1UL) & alignSizeMask;
    }

private:
    uint8_t *base_ = nullptr;
    uint64_t size_ = 0;
    std::map<uint64_t, uint64_t> addressIdleTree_;
    std::map<uint64_t, uint64_t> addressUsedTree_;
    std::set<MemoryRange, RangeSizeFirstComparator> sizeIdleTree_;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_MEM_MANAGER_H