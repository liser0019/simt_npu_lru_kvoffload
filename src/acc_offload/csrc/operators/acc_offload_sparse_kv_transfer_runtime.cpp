/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Descriptor-free Sparse KV transfer for registered Host DVA -> NPU HBM.
 * Thirty-two AIV cores cooperate on every request.  Each core reads the
 * compact miss plan, computes K/V addresses in place, and immediately issues
 * the already validated GM->UB->GM DataCopyPad path.  ResidentAddrs' temporary
 * gvas/addr/size arrays and their global prefix are therefore unnecessary.
 */

#include <cstdint>

#include "kernel_operator.h"

#include "acc_offload_sparse_kv_transfer_runtime.h"

namespace {

constexpr uint32_t TRANSFER_BLOCKS = 32;
constexpr uint32_t TRANSFER_UB_BYTES = 176U * 1024U;

class SparseKvTransferRuntimeKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
        GM_ADDR blockTable, uint64_t hostKBase, uint64_t hostVBase,
        uint64_t deviceKBase, uint64_t deviceVBase, int64_t numReqs,
        int64_t topk, int64_t capacity, int64_t maxNumBlocks,
        int32_t blockSize, int32_t tokenSizeBytesK,
        int32_t tokenSizeBytesV)
    {
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();
        missCount_ = reinterpret_cast<__gm__ int32_t *>(missCount);
        missTokens_ = reinterpret_cast<__gm__ int32_t *>(missTokens);
        missSlots_ = reinterpret_cast<__gm__ int32_t *>(missSlots);
        blockTable_ = reinterpret_cast<__gm__ int32_t *>(blockTable);
        hostKBase_ = hostKBase;
        hostVBase_ = hostVBase;
        deviceKBase_ = deviceKBase;
        deviceVBase_ = deviceVBase;
        numReqs_ = numReqs;
        topk_ = topk;
        capacity_ = capacity;
        maxNumBlocks_ = maxNumBlocks;
        blockSize_ = blockSize;
        tokenSizeBytesK_ = tokenSizeBytesK;
        tokenSizeBytesV_ = tokenSizeBytesV;
        blockBytesK_ = static_cast<int64_t>(blockSize_) * tokenSizeBytesK_;
        blockBytesV_ = static_cast<int64_t>(blockSize_) * tokenSizeBytesV_;
        pipe_.InitBuffer(copyQueue_, 1, TRANSFER_UB_BYTES);
    }

    __aicore__ inline void Process()
    {
        for (int64_t req = 0; req < numReqs_; ++req) {
            int32_t count = missCount_[req];
            if (count < 0) {
                count = 0;
            } else if (static_cast<int64_t>(count) > topk_) {
                count = static_cast<int32_t>(topk_);
            }
            int64_t rowMissBase = req * topk_;
            int64_t rowBlockBase = req * maxNumBlocks_;
            for (int64_t index = aivIndex_; index < count;
                 index += aivNum_) {
                int32_t token = missTokens_[rowMissBase + index];
                int32_t slot = missSlots_[rowMissBase + index];
                if (token < 0 || slot < 0 ||
                    static_cast<int64_t>(slot) >= capacity_) {
                    continue;
                }
                int32_t blockId = token / blockSize_;
                if (blockId < 0 ||
                    static_cast<int64_t>(blockId) >= maxNumBlocks_) {
                    continue;
                }
                int32_t blockIndex = blockTable_[rowBlockBase + blockId];
                if (blockIndex < 0) {
                    continue;
                }
                int32_t offsetInBlock = token % blockSize_;
                uint64_t sourceK = hostKBase_ +
                    static_cast<uint64_t>(blockIndex) * blockBytesK_ +
                    static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesK_;
                uint64_t sourceV = hostVBase_ +
                    static_cast<uint64_t>(blockIndex) * blockBytesV_ +
                    static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesV_;
                uint64_t linearSlot =
                    static_cast<uint64_t>(req) * capacity_ + slot;
                uint64_t destinationK = deviceKBase_ +
                    linearSlot * tokenSizeBytesK_;
                uint64_t destinationV = deviceVBase_ +
                    linearSlot * tokenSizeBytesV_;

                CopyBytes(sourceK, destinationK,
                          static_cast<uint32_t>(tokenSizeBytesK_));
                CopyBytes(sourceV, destinationV,
                          static_cast<uint32_t>(tokenSizeBytesV_));
            }
        }
    }

private:
    __aicore__ inline void CopyBytes(
        uint64_t source, uint64_t destination, uint32_t bytes)
    {
        uint32_t remaining = bytes;
        uint32_t offset = 0;
        AscendC::DataCopyPadExtParams<uint8_t> padParams;
        while (remaining > 0U) {
            uint32_t chunk = remaining > TRANSFER_UB_BYTES
                ? TRANSFER_UB_BYTES
                : remaining;
            AscendC::GlobalTensor<uint8_t> sourceGm;
            AscendC::GlobalTensor<uint8_t> destinationGm;
            sourceGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(source + offset), chunk);
            destinationGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(destination + offset),
                chunk);
            AscendC::LocalTensor<uint8_t> local =
                copyQueue_.AllocTensor<uint8_t>();
            AscendC::DataCopyExtParams copyParams(1, chunk, 0, 0, 0);
            AscendC::DataCopyPad(local, sourceGm, copyParams, padParams);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<uint8_t>();
            AscendC::DataCopyPad(destinationGm, local, copyParams);
            copyQueue_.FreeTensor(local);
            remaining -= chunk;
            offset += chunk;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN,
                      AscendC::TPosition::VECOUT, 1> copyQueue_;
    uint32_t aivNum_ = 0;
    uint32_t aivIndex_ = 0;
    __gm__ int32_t *missCount_ = nullptr;
    __gm__ int32_t *missTokens_ = nullptr;
    __gm__ int32_t *missSlots_ = nullptr;
    __gm__ int32_t *blockTable_ = nullptr;
    uint64_t hostKBase_ = 0;
    uint64_t hostVBase_ = 0;
    uint64_t deviceKBase_ = 0;
    uint64_t deviceVBase_ = 0;
    int64_t numReqs_ = 0;
    int64_t topk_ = 0;
    int64_t capacity_ = 0;
    int64_t maxNumBlocks_ = 0;
    int32_t blockSize_ = 0;
    int32_t tokenSizeBytesK_ = 0;
    int32_t tokenSizeBytesV_ = 0;
    int64_t blockBytesK_ = 0;
    int64_t blockBytesV_ = 0;
};

} // namespace

extern "C" __global__ __aicore__ void SparseKvTransferRuntimeKernelEntry(
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
    GM_ADDR blockTable, uint64_t hostKBase, uint64_t hostVBase,
    uint64_t deviceKBase, uint64_t deviceVBase, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxNumBlocks,
    int32_t blockSize, int32_t tokenSizeBytesK, int32_t tokenSizeBytesV)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SparseKvTransferRuntimeKernel kernel;
    kernel.Init(
        missCount, missTokens, missSlots, blockTable, hostKBase, hostVBase,
        deviceKBase, deviceVBase, numReqs, topk, capacity, maxNumBlocks,
        blockSize, tokenSizeBytesK, tokenSizeBytesV);
    kernel.Process();
}

void OffloadOpsSparseKvTransferRuntime(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t block_table, uint64_t host_k_base, uint64_t host_v_base,
    uint64_t device_k_base, uint64_t device_v_base, int64_t num_reqs,
    int64_t topk, int64_t capacity, int64_t max_num_blocks,
    int32_t block_size, int32_t token_size_bytes_k,
    int32_t token_size_bytes_v, void *stream)
{
    if (num_reqs <= 0 || topk <= 0 || capacity <= 0 ||
        max_num_blocks <= 0 || block_size <= 0 ||
        token_size_bytes_k <= 0 || token_size_bytes_v <= 0) {
        return;
    }
    SparseKvTransferRuntimeKernelEntry<<<TRANSFER_BLOCKS, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(miss_count),
        reinterpret_cast<GM_ADDR>(miss_tokens),
        reinterpret_cast<GM_ADDR>(miss_slots),
        reinterpret_cast<GM_ADDR>(block_table), host_k_base, host_v_base,
        device_k_base, device_v_base, num_reqs, topk, capacity,
        max_num_blocks, block_size, token_size_bytes_k,
        token_size_bytes_v);
}
