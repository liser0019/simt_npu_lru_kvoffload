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

#ifndef ACC_OFFLOAD_SPARSE_COPY_H
#define ACC_OFFLOAD_SPARSE_COPY_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr int64_t UB_ONCE_SIZE = 176 * 1024;

template<typename T>
class OffloadSparseCopyKernel {
public:
    HYBM_AICORE_KERNEL OffloadSparseCopyKernel() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
    {
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();

        uint32_t oriSize = *(reinterpret_cast<__gm__ uint32_t *>(size));
        size_ = oriSize / 2;
        inputs_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
        outputs_ = reinterpret_cast<__gm__ uint64_t *>(outputs);
        lens_ = reinterpret_cast<__gm__ uint32_t *>(lens);
        pipe_.InitBuffer(bindQueue_, 1, UB_ONCE_SIZE);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        uint32_t perCoreSize = size_ / aivNum_;
        uint32_t kStart = aivIndex_ * perCoreSize;
        uint32_t vStart = size_ + kStart;
        uint32_t lastAivIdx = aivNum_ - 1;
        if (aivIndex_ == lastAivIdx) {
            perCoreSize = size_ - lastAivIdx * perCoreSize;
        }
        uint32_t kEnd = kStart + perCoreSize;
        uint32_t vEnd = vStart + perCoreSize;

        for (uint32_t i = kStart; i < kEnd; i++) {
            auto inputPtr = reinterpret_cast<__gm__ T *>(inputs_[i]);
            auto outputPtr = reinterpret_cast<__gm__ T *>(outputs_[i]);
            auto len = lens_[i];
            inputGm_.SetGlobalBuffer(inputPtr, len);
            outputGm_.SetGlobalBuffer(outputPtr, len);
            CpGM2GM(len);
        }

        for (uint32_t i = vStart; i < vEnd; i++) {
            auto inputPtr = reinterpret_cast<__gm__ T *>(inputs_[i]);
            auto outputPtr = reinterpret_cast<__gm__ T *>(outputs_[i]);
            auto len = lens_[i];
            inputGm_.SetGlobalBuffer(inputPtr, len);
            outputGm_.SetGlobalBuffer(outputPtr, len);
            CpGM2GM(len);
        }
    }

private:
    HYBM_AICORE_KERNEL void CpGM2GM(uint32_t len)
    {
        uint32_t leftLen = len * sizeof(T);
        uint32_t times = 0;
        uint32_t preCopyNum = UB_ONCE_SIZE / sizeof(T);
        AscendC::DataCopyPadExtParams<T> padParams;

        while (leftLen > 0) {
            uint32_t curCopySize = (leftLen > UB_ONCE_SIZE) ? UB_ONCE_SIZE : leftLen;
            AscendC::LocalTensor<T> local = bindQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyParams(1, curCopySize, 0, 0, 0);
            AscendC::DataCopyPad(local, inputGm_[times * preCopyNum], dataCopyParams, padParams);
            bindQueue_.EnQue(local);
            local = bindQueue_.DeQue<T>();
            AscendC::DataCopyPad(outputGm_[times * preCopyNum], local, dataCopyParams);
            bindQueue_.FreeTensor(local);
            leftLen = (leftLen > UB_ONCE_SIZE) ? leftLen - UB_ONCE_SIZE : 0;
            times++;
        };

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> bindQueue_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint32_t aivNum_;
    uint32_t aivIndex_;
    uint32_t size_;
    __gm__ uint64_t *inputs_;
    __gm__ uint64_t *outputs_;
    __gm__ uint32_t *lens_;
};

#endif // ACC_OFFLOAD_SPARSE_COPY_H