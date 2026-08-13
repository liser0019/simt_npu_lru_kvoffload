/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Parallel V1 for ComputeLruResidentAddrs on the production num_reqs=1 path.
 *
 * The legacy pure-SIMT kernel maps one lane to one request.  With the real
 * num_reqs=1 workload that leaves lane 0 serially processing as many as 2048
 * misses while the other 127 lanes return.  This mixed kernel instead launches
 * one 1024-lane VF and maps misses, not requests, to lanes.  Lane t owns input
 * positions t and t+1024.
 *
 * Each input is loaded, validated and translated through block_table exactly
 * once.  Its small metadata remains in scalar registers while two deterministic
 * warp-prefix scans establish the stable packed ranks and final valid count.
 * We cannot publish V immediately after the first tile because the ABI is
 * [all K][all V] and the V base is the final valid count across both tiles.
 *
 * Only 288 bytes of dynamic UB are used for warp totals/bases and control.
 * Descriptor input/output stays direct GM in this V1 so the experiment isolates
 * task mapping, duplicate-scan removal and warp-prefix compaction.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace {

constexpr uint32_t RESIDENT_ADDRS_THREADS = 1024;
constexpr uint32_t RESIDENT_ADDRS_WARP_SIZE = 32;
constexpr uint32_t RESIDENT_ADDRS_WARP_COUNT =
    RESIDENT_ADDRS_THREADS / RESIDENT_ADDRS_WARP_SIZE;
constexpr uint32_t RESIDENT_ADDRS_MAX_TOPK = 2048;

// The two scans are sequential and therefore reuse the same 32-entry arrays.
// Keep the control tail eight-int (32-byte) aligned for explicit dynamic UB.
constexpr uint32_t WARP_TOTALS_OFFSET = 0;
constexpr uint32_t WARP_BASES_OFFSET =
    WARP_TOTALS_OFFSET + RESIDENT_ADDRS_WARP_COUNT;
constexpr uint32_t CONTROL_OFFSET =
    WARP_BASES_OFFSET + RESIDENT_ADDRS_WARP_COUNT;
constexpr uint32_t CONTROL_COUNT = 0;
constexpr uint32_t CONTROL_TOTAL = 1;
constexpr uint32_t CONTROL_ELEMENTS = 8;
constexpr uint32_t RESIDENT_ADDRS_DYNAMIC_UB_ELEMENTS =
    CONTROL_OFFSET + CONTROL_ELEMENTS;
constexpr uint32_t RESIDENT_ADDRS_DYNAMIC_UB_BYTES =
    RESIDENT_ADDRS_DYNAMIC_UB_ELEMENTS * sizeof(int32_t);

static_assert(RESIDENT_ADDRS_WARP_COUNT == 32U,
              "1024 lanes must form 32 A5 warps");
static_assert(RESIDENT_ADDRS_DYNAMIC_UB_BYTES == 288U,
              "ResidentAddrs scan scratch layout changed unexpectedly");
static_assert(RESIDENT_ADDRS_DYNAMIC_UB_BYTES % 32U == 0U,
              "dynamic UB must remain 32-byte aligned");

// Work-efficient 1024-lane stable exclusive scan for one 0/1 predicate.
// Five shuffle stages scan each 32-lane warp.  Lane 31 publishes the warp
// total; warp 0 scans those 32 totals; every lane then adds its warp base.
// All lanes execute all three block barriers, including inactive input lanes.
__simt_callee__ __aicore__ inline int32_t WarpExclusiveScan1024(
    __ubuf__ int32_t *workspace, int32_t flag, uint32_t thread)
{
    __ubuf__ int32_t *warpTotals = workspace + WARP_TOTALS_OFFSET;
    __ubuf__ int32_t *warpBases = workspace + WARP_BASES_OFFSET;
    __ubuf__ int32_t *control = workspace + CONTROL_OFFSET;
    uint32_t lane = thread & (RESIDENT_ADDRS_WARP_SIZE - 1U);
    uint32_t warp = thread / RESIDENT_ADDRS_WARP_SIZE;

    int32_t inclusive = flag;
    for (uint32_t offset = 1; offset < RESIDENT_ADDRS_WARP_SIZE;
         offset <<= 1) {
        int32_t prior =
            asc_shfl_up(inclusive, offset, RESIDENT_ADDRS_WARP_SIZE);
        if (lane >= offset) {
            inclusive += prior;
        }
    }
    if (lane == RESIDENT_ADDRS_WARP_SIZE - 1U) {
        warpTotals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t warpValue = warpTotals[lane];
        int32_t warpInclusive = warpValue;
        for (uint32_t offset = 1; offset < RESIDENT_ADDRS_WARP_COUNT;
             offset <<= 1) {
            int32_t prior = asc_shfl_up(
                warpInclusive, offset, RESIDENT_ADDRS_WARP_SIZE);
            if (lane >= offset) {
                warpInclusive += prior;
            }
        }
        warpBases[lane] = warpInclusive - warpValue;
    }
    asc_syncthreads();

    int32_t exclusive = warpBases[warp] + inclusive - flag;
    if (thread == 0U) {
        control[CONTROL_TOTAL] =
            warpBases[RESIDENT_ADDRS_WARP_COUNT - 1U] +
            warpTotals[RESIDENT_ADDRS_WARP_COUNT - 1U];
    }
    asc_syncthreads();
    return exclusive;
}

__simt_callee__ __aicore__ inline void PublishResidentAddressItem(
    int32_t rank, int32_t totalValid, int32_t slot, int32_t blockIndex,
    int32_t offsetInBlock, int64_t blockBytesK, int64_t blockBytesV,
    __gm__ int64_t *gvasBuffer,
    __gm__ int64_t *addrBuffer, __gm__ int32_t *sizeBuffer,
    int32_t tokenSizeBytesK, int32_t tokenSizeBytesV, int64_t gvasKBase,
    int64_t gvasVBase, int64_t addrKBase, int64_t addrVBase)
{
    int64_t kPos = rank;
    int64_t vPos = static_cast<int64_t>(totalValid) + rank;
    int64_t itemOffset = static_cast<int64_t>(offsetInBlock);

    gvasBuffer[kPos] =
        gvasKBase + static_cast<int64_t>(blockIndex) * blockBytesK +
        itemOffset * tokenSizeBytesK;
    gvasBuffer[vPos] =
        gvasVBase + static_cast<int64_t>(blockIndex) * blockBytesV +
        itemOffset * tokenSizeBytesV;
    addrBuffer[kPos] =
        addrKBase + static_cast<int64_t>(slot) * tokenSizeBytesK;
    addrBuffer[vPos] =
        addrVBase + static_cast<int64_t>(slot) * tokenSizeBytesV;
    sizeBuffer[kPos] = tokenSizeBytesK;
    sizeBuffer[vPos] = tokenSizeBytesV;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(RESIDENT_ADDRS_THREADS) inline void
ResidentAddrsParallelVf(
    __ubuf__ int32_t *workspace, __gm__ int32_t *missCount,
    __gm__ int32_t *missTokens, __gm__ int32_t *missSlots,
    __gm__ int32_t *blockTable, __gm__ int64_t *gvasBuffer,
    __gm__ int64_t *addrBuffer, __gm__ int32_t *sizeBuffer,
    __gm__ int32_t *numTokensBuffer, int32_t blockSize,
    int32_t tokenSizeBytesK, int32_t tokenSizeBytesV, int64_t gvasKBase,
    int64_t gvasVBase, int64_t addrKBase, int64_t addrVBase,
    int32_t residentCapacity, int64_t topk, int64_t maxNumBlocks)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *control = workspace + CONTROL_OFFSET;

    if (thread == 0U) {
        int32_t count = 0;
        if (topk > 0 && topk <= RESIDENT_ADDRS_MAX_TOPK && blockSize > 0 &&
            residentCapacity > 0 && maxNumBlocks > 0) {
            count = *missCount;
            if (count < 0) {
                count = 0;
            } else if (count > topk) {
                count = static_cast<int32_t>(topk);
            }
        }
        control[CONTROL_COUNT] = count;
    }
    asc_syncthreads();
    int32_t count = control[CONTROL_COUNT];

    // No lane may return before the scans: even count=0 must converge at every
    // block barrier.  Inactive items simply contribute flag=0.
    uint32_t index0 = thread;
    uint32_t index1 = thread + RESIDENT_ADDRS_THREADS;
    // Keep two independent register records.  The validation code is expanded
    // here (rather than passing scalar references through a SIMT callee) to
    // stay within the Bisheng calling convention already proven by this repo.
    int32_t valid0 = 0;
    int32_t slot0 = 0;
    int32_t blockIndex0 = 0;
    int32_t offsetInBlock0 = 0;
    if (index0 < static_cast<uint32_t>(count)) {
        // Keep token/block-id temporaries inside validation so Bisheng can end
        // their live ranges before the two scans.  Only compact int32 metadata
        // survives across both block-wide prefix operations.
        const int32_t token = missTokens[index0];
        const int32_t candidateSlot = missSlots[index0];
        if (token >= 0 && candidateSlot >= 0 &&
            candidateSlot < residentCapacity) {
            const int32_t blockId = token / blockSize;
            if (static_cast<int64_t>(blockId) < maxNumBlocks) {
                const int32_t candidateBlock = blockTable[blockId];
                if (candidateBlock >= 0) {
                    valid0 = 1;
                    slot0 = candidateSlot;
                    blockIndex0 = candidateBlock;
                    offsetInBlock0 = token % blockSize;
                }
            }
        }
    }

    int32_t valid1 = 0;
    int32_t slot1 = 0;
    int32_t blockIndex1 = 0;
    int32_t offsetInBlock1 = 0;
    if (index1 < static_cast<uint32_t>(count)) {
        const int32_t token = missTokens[index1];
        const int32_t candidateSlot = missSlots[index1];
        if (token >= 0 && candidateSlot >= 0 &&
            candidateSlot < residentCapacity) {
            const int32_t blockId = token / blockSize;
            if (static_cast<int64_t>(blockId) < maxNumBlocks) {
                const int32_t candidateBlock = blockTable[blockId];
                if (candidateBlock >= 0) {
                    valid1 = 1;
                    slot1 = candidateSlot;
                    blockIndex1 = candidateBlock;
                    offsetInBlock1 = token % blockSize;
                }
            }
        }
    }

    int32_t rank0 = WarpExclusiveScan1024(
        workspace, valid0, thread);
    int32_t total0 = control[CONTROL_TOTAL];
    int32_t rank1Local = WarpExclusiveScan1024(
        workspace, valid1, thread);
    int32_t totalValid = total0 + control[CONTROL_TOTAL];
    int32_t rank1 = total0 + rank1Local;

    if (thread == 0U) {
        *numTokensBuffer = totalValid * 2;
    }
    if (valid0 != 0 || valid1 != 0) {
        // These values are uniform and needed only for final publication.
        // Compute them once per publishing lane after both scans instead of
        // once for each item, and do not keep four full 64-bit addresses live
        // across the barriers.  No block barrier follows this lane-local
        // branch, so inactive lanes may safely skip it.
        const int64_t blockBytesK =
            static_cast<int64_t>(blockSize) * tokenSizeBytesK;
        const int64_t blockBytesV =
            static_cast<int64_t>(blockSize) * tokenSizeBytesV;
        if (valid0 != 0) {
            PublishResidentAddressItem(
                rank0, totalValid, slot0, blockIndex0, offsetInBlock0,
                blockBytesK, blockBytesV, gvasBuffer, addrBuffer, sizeBuffer,
                tokenSizeBytesK, tokenSizeBytesV, gvasKBase, gvasVBase,
                addrKBase, addrVBase);
        }
        if (valid1 != 0) {
            PublishResidentAddressItem(
                rank1, totalValid, slot1, blockIndex1, offsetInBlock1,
                blockBytesK, blockBytesV, gvasBuffer, addrBuffer, sizeBuffer,
                tokenSizeBytesK, tokenSizeBytesV, gvasKBase, gvasVBase,
                addrKBase, addrVBase);
        }
    }
}

} // namespace

extern "C" __global__ __vector__ void
OffloadComputeLruResidentAddrsMixedParallelKernel(
    GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
    GM_ADDR blockTable, GM_ADDR gvasBuffer, GM_ADDR addrBuffer,
    GM_ADDR sizeBuffer, GM_ADDR numTokensBuffer, int32_t blockSize,
    int32_t tokenSizeBytesK, int32_t tokenSizeBytesV, int64_t gvasKBase,
    int64_t gvasVBase, int64_t addrKBase, int64_t addrVBase,
    int32_t residentCapacity, int64_t topk, int64_t maxNumBlocks)
{
    AscendC::InitSocState();
    extern __ubuf__ int32_t workspace[];
    asc_vf_call<ResidentAddrsParallelVf>(
        dim3(RESIDENT_ADDRS_THREADS), workspace,
        reinterpret_cast<__gm__ int32_t *>(missCount),
        reinterpret_cast<__gm__ int32_t *>(missTokens),
        reinterpret_cast<__gm__ int32_t *>(missSlots),
        reinterpret_cast<__gm__ int32_t *>(blockTable),
        reinterpret_cast<__gm__ int64_t *>(gvasBuffer),
        reinterpret_cast<__gm__ int64_t *>(addrBuffer),
        reinterpret_cast<__gm__ int32_t *>(sizeBuffer),
        reinterpret_cast<__gm__ int32_t *>(numTokensBuffer), blockSize,
        tokenSizeBytesK, tokenSizeBytesV, gvasKBase, gvasVBase, addrKBase,
        addrVBase, residentCapacity, topk, maxNumBlocks);
}

extern "C" void OffloadOpsComputeLruResidentAddrsMixedParallel(
    uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
    uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
    uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
    int32_t token_size_bytes_k, int32_t token_size_bytes_v,
    int64_t gvas_k_base, int64_t gvas_v_base, int64_t addr_k_base,
    int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs,
    int64_t topk, int64_t max_num_blocks, void *stream)
{
    // Host dispatch shape-gates this backend. Keep a defensive guard because
    // this internal symbol can still be inspected or called independently.
    if (num_reqs != 1 || topk <= 0 || topk > RESIDENT_ADDRS_MAX_TOPK) {
        return;
    }
    OffloadComputeLruResidentAddrsMixedParallelKernel<<<
        1, RESIDENT_ADDRS_DYNAMIC_UB_BYTES, stream>>>(
        reinterpret_cast<GM_ADDR>(miss_count),
        reinterpret_cast<GM_ADDR>(miss_tokens),
        reinterpret_cast<GM_ADDR>(miss_slots),
        reinterpret_cast<GM_ADDR>(block_table),
        reinterpret_cast<GM_ADDR>(gvas_buffer),
        reinterpret_cast<GM_ADDR>(addr_buffer),
        reinterpret_cast<GM_ADDR>(size_buffer),
        reinterpret_cast<GM_ADDR>(num_tokens_buffer), block_size,
        token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base,
        addr_k_base, addr_v_base, resident_capacity, topk, max_num_blocks);
}
