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
#ifndef ACC_OFFLOAD_LRU_RESIDENT_ADDRS_H
#define ACC_OFFLOAD_LRU_RESIDENT_ADDRS_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr int64_t LRU_ADDRS_UB_ALIGN = 32;
constexpr uint32_t LRU_ADDRS_BLOCK_DIM = 8;

class ComputeLruResidentAddrsKernel {
public:
    HYBM_AICORE_KERNEL ComputeLruResidentAddrsKernel() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR miss_count, GM_ADDR miss_tokens, GM_ADDR miss_slots, GM_ADDR block_table,
                                 GM_ADDR gvas_buffer, GM_ADDR addr_buffer, GM_ADDR size_buffer,
                                 GM_ADDR num_tokens_buffer, int32_t block_size, int32_t token_size_bytes_k,
                                 int32_t token_size_bytes_v, int64_t gvas_k_base, int64_t gvas_v_base,
                                 int64_t addr_k_base, int64_t addr_v_base, int32_t resident_capacity,
                                 int64_t num_reqs, int64_t topk, int64_t max_num_blocks)
    {
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();
        num_reqs_ = num_reqs;
        topk_ = topk;
        max_num_blocks_ = max_num_blocks;
        block_size_ = block_size;
        token_size_bytes_k_ = token_size_bytes_k;
        token_size_bytes_v_ = token_size_bytes_v;
        gvas_k_base_ = gvas_k_base;
        gvas_v_base_ = gvas_v_base;
        addr_k_base_ = addr_k_base;
        addr_v_base_ = addr_v_base;
        resident_capacity_ = resident_capacity;

        miss_count_ = reinterpret_cast<__gm__ int32_t *>(miss_count);
        miss_tokens_ = reinterpret_cast<__gm__ int32_t *>(miss_tokens);
        miss_slots_ = reinterpret_cast<__gm__ int32_t *>(miss_slots);
        block_table_ = reinterpret_cast<__gm__ int32_t *>(block_table);
        gvas_buffer_ = reinterpret_cast<__gm__ int64_t *>(gvas_buffer);
        addr_buffer_ = reinterpret_cast<__gm__ int64_t *>(addr_buffer);
        size_buffer_ = reinterpret_cast<__gm__ int32_t *>(size_buffer);
        num_tokens_buffer_ = reinterpret_cast<__gm__ int32_t *>(num_tokens_buffer);

        uint32_t reqsBytes = AlignUpU32(static_cast<uint32_t>(num_reqs) * sizeof(int32_t));
        pipe_.InitBuffer(missCountBuf_, reqsBytes);
        pipe_.InitBuffer(reqStartBuf_, reqsBytes);

        // Per-req UB staging chunks for the 6 output buffers. Sized by topk
        // (upper bound on missCnt per req). Total UB: topk * (4*8 + 2*4) bytes.
        uint32_t chunkBytes64 = AlignUpU32(static_cast<uint32_t>(topk) * sizeof(int64_t));
        uint32_t chunkBytes32 = AlignUpU32(static_cast<uint32_t>(topk) * sizeof(int32_t));
        pipe_.InitBuffer(gvasKBuf_, chunkBytes64);
        pipe_.InitBuffer(gvasVBuf_, chunkBytes64);
        pipe_.InitBuffer(addrKBuf_, chunkBytes64);
        pipe_.InitBuffer(addrVBuf_, chunkBytes64);
        pipe_.InitBuffer(sizeKBuf_, chunkBytes32);
        pipe_.InitBuffer(sizeVBuf_, chunkBytes32);
        pipe_.InitBuffer(numTokensBuf_, static_cast<uint32_t>(LRU_ADDRS_UB_ALIGN));
    }

    HYBM_AICORE_KERNEL void Process()
    {
        // BuildPrefixSum
        auto missCountLocal = missCountBuf_.Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(num_reqs_ * sizeof(int32_t)), 0, 0, 0};
        AscendC::GlobalTensor<int32_t> missCountGm;
        missCountGm.SetGlobalBuffer(miss_count_, static_cast<uint64_t>(num_reqs_));
        AscendC::DataCopyPad(missCountLocal, missCountGm, cp, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t cumsum = 0;
        int32_t topk = topk_;
        auto reqStartLocal = reqStartBuf_.Get<int32_t>();
        for (int64_t r = 0; r < num_reqs_; ++r) {
            reqStartLocal.SetValue(r, cumsum);
            int32_t cnt = missCountLocal.GetValue(r);
            if (cnt < 0) {
                cnt = 0;
            } else if (cnt > topk) {
                cnt = topk;
            }
            cumsum += cnt;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if (aivIndex_ == 0) {
            StoreNumTokens(cumsum * 2);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // EmitAddrs
        auto gvasKLocal = gvasKBuf_.Get<int64_t>();
        auto gvasVLocal = gvasVBuf_.Get<int64_t>();
        auto addrKLocal = addrKBuf_.Get<int64_t>();
        auto addrVLocal = addrVBuf_.Get<int64_t>();
        auto sizeKLocal = sizeKBuf_.Get<int32_t>();
        auto sizeVLocal = sizeVBuf_.Get<int32_t>();

        int32_t blockSize = block_size_;
        int32_t blockSizeBytesK = blockSize * token_size_bytes_k_;
        int32_t blockSizeBytesV = blockSize * token_size_bytes_v_;
        int32_t cap = resident_capacity_;
        int32_t maxBlocks = max_num_blocks_;

        for (int64_t req = aivIndex_; req < num_reqs_; req += aivNum_) {
            int32_t missCnt = missCountLocal.GetValue(req);
            if (missCnt <= 0) {
                continue;
            }
            int32_t reqStart = reqStartLocal.GetValue(req);
            for (int32_t idx = 0; idx < missCnt; ++idx) {
                int32_t token = miss_tokens_[req * topk + idx];
                int32_t slot = miss_slots_[req * topk + idx];
                if (token < 0 || slot < 0 || slot >= cap) {
                    continue;
                }
                int32_t blockId = token / blockSize;
                if (blockId < 0 || blockId >= maxBlocks) {
                    continue;
                }

                int64_t offsetInBlock = token % blockSize;
                int64_t blockIndice = block_table_[req * maxBlocks + blockId];
                int64_t gvas_k = gvas_k_base_ + blockIndice * blockSizeBytesK + offsetInBlock * token_size_bytes_k_;
                int64_t gvas_v = gvas_v_base_ + blockIndice * blockSizeBytesV + offsetInBlock * token_size_bytes_v_;
                int64_t addr_k = addr_k_base_ + (req * cap + slot) * token_size_bytes_k_;
                int64_t addr_v = addr_v_base_ + (req * cap + slot) * token_size_bytes_v_;

                gvasKLocal.SetValue(idx, gvas_k);
                gvasVLocal.SetValue(idx, gvas_v);
                addrKLocal.SetValue(idx, addr_k);
                addrVLocal.SetValue(idx, addr_v);
                sizeKLocal.SetValue(idx, token_size_bytes_k_);
                sizeVLocal.SetValue(idx, token_size_bytes_v_);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            uint32_t bytes64 = missCnt * sizeof(int64_t);
            uint32_t bytes32 = missCnt * sizeof(int32_t);
            AscendC::DataCopyExtParams cp64{1, bytes64, 0, 0, 0};
            AscendC::DataCopyExtParams cp32{1, bytes32, 0, 0, 0};

            int32_t posKBase = reqStart;
            int32_t posVBase = cumsum + reqStart;

            AscendC::GlobalTensor<int64_t> gvasKGm;
            gvasKGm.SetGlobalBuffer(gvas_buffer_ + posKBase, missCnt);
            AscendC::DataCopyPad(gvasKGm, gvasKLocal, cp64);

            AscendC::GlobalTensor<int64_t> gvasVGm;
            gvasVGm.SetGlobalBuffer(gvas_buffer_ + posVBase, missCnt);
            AscendC::DataCopyPad(gvasVGm, gvasVLocal, cp64);

            AscendC::GlobalTensor<int64_t> addrKGm;
            addrKGm.SetGlobalBuffer(addr_buffer_ + posKBase, missCnt);
            AscendC::DataCopyPad(addrKGm, addrKLocal, cp64);

            AscendC::GlobalTensor<int64_t> addrVGm;
            addrVGm.SetGlobalBuffer(addr_buffer_ + posVBase, missCnt);
            AscendC::DataCopyPad(addrVGm, addrVLocal, cp64);

            AscendC::GlobalTensor<int32_t> sizeKGm;
            sizeKGm.SetGlobalBuffer(size_buffer_ + posKBase, missCnt);
            AscendC::DataCopyPad(sizeKGm, sizeKLocal, cp32);

            AscendC::GlobalTensor<int32_t> sizeVGm;
            sizeVGm.SetGlobalBuffer(size_buffer_ + posVBase, missCnt);
            AscendC::DataCopyPad(sizeVGm, sizeVLocal, cp32);

            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    HYBM_AICORE_KERNEL uint32_t AlignUpU32(uint32_t bytes)
    {
        return (bytes + static_cast<uint32_t>(LRU_ADDRS_UB_ALIGN) - 1) &
               ~(static_cast<uint32_t>(LRU_ADDRS_UB_ALIGN) - 1);
    }

    HYBM_AICORE_KERNEL void StoreNumTokens(int32_t value)
    {
        auto stage = numTokensBuf_.Get<int32_t>();
        stage.SetValue(0, value);
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::GlobalTensor<int32_t> gm;
        gm.SetGlobalBuffer(num_tokens_buffer_, 1);
        AscendC::DataCopyPad(gm, stage, cp);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<> missCountBuf_;
    AscendC::TBuf<> reqStartBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gvasKBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gvasVBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> addrKBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> addrVBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sizeKBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sizeVBuf_;
    AscendC::TBuf<> numTokensBuf_;

    uint32_t aivNum_ = 0;
    uint32_t aivIndex_ = 0;
    int64_t num_reqs_ = 0;
    int64_t topk_ = 0;
    int64_t max_num_blocks_ = 0;
    int32_t block_size_ = 0;
    int32_t token_size_bytes_k_ = 0;
    int32_t token_size_bytes_v_ = 0;
    int64_t gvas_k_base_ = 0;
    int64_t gvas_v_base_ = 0;
    int64_t addr_k_base_ = 0;
    int64_t addr_v_base_ = 0;
    int32_t resident_capacity_ = 0;

    __gm__ int32_t *miss_count_ = nullptr;
    __gm__ int32_t *miss_tokens_ = nullptr;
    __gm__ int32_t *miss_slots_ = nullptr;
    __gm__ int32_t *block_table_ = nullptr;
    __gm__ int64_t *gvas_buffer_ = nullptr;
    __gm__ int64_t *addr_buffer_ = nullptr;
    __gm__ int32_t *size_buffer_ = nullptr;
    __gm__ int32_t *num_tokens_buffer_ = nullptr;
};

#endif // ACC_OFFLOAD_LRU_RESIDENT_ADDRS_H
