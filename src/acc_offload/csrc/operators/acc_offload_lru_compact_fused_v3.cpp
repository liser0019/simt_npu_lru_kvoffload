/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Production Compact V3 implementation.
 *
 * This public-sharing source keeps only the A5-validated direct-GM V3 path:
 * one mixed kernel, one 1024-lane SIMT VF, a 114720-byte dynamic-UB workspace,
 * deterministic UB atomics, warp-prefix compaction, the reset fast path and
 * the all-hit continuation.  Historical V1/V2 and the failed MTE3 publication
 * experiment have intentionally been removed from this translation unit.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_warp_functions.h"
#include "utils/debug/asc_time.h"

namespace {

constexpr uint32_t LRU_FUSED_V3_THREADS = 1024;
constexpr uint32_t LRU_FUSED_V3_WARP_SIZE = 32;
constexpr uint32_t LRU_FUSED_V3_WARP_COUNT =
    LRU_FUSED_V3_THREADS / LRU_FUSED_V3_WARP_SIZE;
constexpr uint32_t LRU_FUSED_V3_TOPK = 2048;
constexpr uint32_t LRU_FUSED_V3_CAPACITY = 4096;
constexpr uint32_t LRU_FUSED_V3_MAX_TOKEN = 4096;
constexpr int64_t LRU_FUSED_V3_WORKSPACE_THREADS = 8;
constexpr int32_t LRU_INVALID_TOKEN = -1;
constexpr int32_t LRU_FUSED_V3_EPOCH = -2;

constexpr uint32_t LRU_FUSED_V3_TOPK_BYTES =
    LRU_FUSED_V3_TOPK * sizeof(int32_t);
constexpr uint32_t LRU_FUSED_V3_CAPACITY_BYTES =
    LRU_FUSED_V3_CAPACITY * sizeof(int32_t);
constexpr uint32_t LRU_FUSED_V3_TOPK_BLOCKS =
    LRU_FUSED_V3_TOPK_BYTES / 32U;
constexpr uint32_t LRU_FUSED_V3_CAPACITY_BLOCKS =
    LRU_FUSED_V3_CAPACITY_BYTES / 32U;
constexpr uint32_t LRU_FUSED_V3_GM_INPUT_BYTES =
    LRU_FUSED_V3_TOPK_BYTES + 2U * LRU_FUSED_V3_CAPACITY_BYTES;
constexpr uint32_t LRU_FUSED_V3_GM_OUTPUT_BYTES =
    2U * LRU_FUSED_V3_CAPACITY_BYTES +
    3U * LRU_FUSED_V3_TOPK_BYTES + sizeof(int64_t) + 2U * sizeof(int32_t);

// V3 preserves the A5-validated 114720-byte UB layout.  The historic V2 names
// in old branches described the layout's origin; this cleaned source treats it
// simply as the production V3 layout.
constexpr uint32_t TOPK_OFFSET = 0;
constexpr uint32_t SLOT_TO_TOKEN_OFFSET = TOPK_OFFSET + LRU_FUSED_V3_TOPK;
constexpr uint32_t LRU_SLOTS_OFFSET =
    SLOT_TO_TOKEN_OFFSET + LRU_FUSED_V3_CAPACITY;
constexpr uint32_t CURRENT_SLOTS_OFFSET =
    LRU_SLOTS_OFFSET + LRU_FUSED_V3_CAPACITY;
constexpr uint32_t MISS_TOKENS_OFFSET =
    CURRENT_SLOTS_OFFSET + LRU_FUSED_V3_TOPK;
constexpr uint32_t TOKEN_MARK_OFFSET =
    MISS_TOKENS_OFFSET + LRU_FUSED_V3_TOPK;
constexpr uint32_t TOKEN_POS_OFFSET =
    TOKEN_MARK_OFFSET + LRU_FUSED_V3_MAX_TOKEN;
constexpr uint32_t GOLDEN_PAYLOAD_ELEMENTS =
    TOKEN_POS_OFFSET + LRU_FUSED_V3_MAX_TOKEN;
constexpr uint32_t GOLDEN_PAYLOAD_BYTES =
    GOLDEN_PAYLOAD_ELEMENTS * sizeof(int32_t);
constexpr uint32_t MISS_SLOTS_OFFSET = GOLDEN_PAYLOAD_ELEMENTS;
constexpr uint32_t SCAN_EVICT_A_OFFSET =
    MISS_SLOTS_OFFSET + LRU_FUSED_V3_TOPK;
constexpr uint32_t SCAN_EVICT_B_OFFSET =
    SCAN_EVICT_A_OFFSET + LRU_FUSED_V3_THREADS;
constexpr uint32_t SCAN_HIT_A_OFFSET =
    SCAN_EVICT_B_OFFSET + LRU_FUSED_V3_THREADS;
constexpr uint32_t SCAN_HIT_B_OFFSET =
    SCAN_HIT_A_OFFSET + LRU_FUSED_V3_THREADS;
constexpr uint32_t SCAN_CONTROL_OFFSET =
    SCAN_HIT_B_OFFSET + LRU_FUSED_V3_THREADS;

constexpr uint32_t CONTROL_CURRENT_EPOCH = 0;
constexpr uint32_t CONTROL_EVICTABLE_COUNT = 1;
constexpr uint32_t CONTROL_HIT_COUNT = 2;
constexpr uint32_t CONTROL_MISS_COUNT = 3;
constexpr uint32_t CONTROL_RESET_ROW = 4;
constexpr uint32_t CONTROL_STABLE_PREFIX = 5;
constexpr uint32_t CONTROL_HAS_MISS = 6;
constexpr uint32_t CONTROL_ELEMENTS_ALIGNED = 8;

constexpr uint32_t SCAN_EVICT_RESULT_OFFSET = SCAN_EVICT_A_OFFSET;
constexpr uint32_t SCAN_HIT_RESULT_OFFSET = SCAN_HIT_A_OFFSET;
constexpr uint32_t FUSED_V3_DYNAMIC_UB_ELEMENTS =
    SCAN_CONTROL_OFFSET + CONTROL_ELEMENTS_ALIGNED;
constexpr uint32_t FUSED_V3_DYNAMIC_UB_BYTES =
    FUSED_V3_DYNAMIC_UB_ELEMENTS * sizeof(int32_t);

constexpr uint32_t V3_PROFILE_STAMPS = 8;

static_assert(LRU_FUSED_V3_WARP_COUNT == 32U,
              "1024 lanes must form 32 A5 warps");
static_assert(GOLDEN_PAYLOAD_BYTES == 88U * 1024U,
              "V3 must preserve the seven V1 offsets");
static_assert(MISS_SLOTS_OFFSET * sizeof(int32_t) == 88U * 1024U,
              "V3 miss_slots must preserve the V1 offset");
static_assert(FUSED_V3_DYNAMIC_UB_BYTES == 114720U,
              "V3 must preserve the validated dynamic UB allocation");
static_assert(FUSED_V3_DYNAMIC_UB_BYTES % 32U == 0,
              "V3 dynamic UB must be 32-byte aligned");
static_assert(LRU_FUSED_V3_GM_INPUT_BYTES == 40960U,
              "V3 large-array input must be exactly 40 KiB");
static_assert(LRU_FUSED_V3_GM_OUTPUT_BYTES == 57360U,
              "V3 external publication size changed unexpectedly");
// A5 warp size is 32. Each warp performs a five-stage register shuffle scan,
// lane 31 publishes its two category counts, and warp 0 scans those 32 counts.
__simt_callee__ __aicore__ inline void WarpScanPair1024(
    __ubuf__ int32_t *evictA, __ubuf__ int32_t *evictB,
    __ubuf__ int32_t *hitA, __ubuf__ int32_t *hitB,
    int32_t evictFlag, int32_t hitFlag, uint32_t thread)
{
    uint32_t lane = thread & (LRU_FUSED_V3_WARP_SIZE - 1U);
    uint32_t warp = thread / LRU_FUSED_V3_WARP_SIZE;
    int32_t evictInclusive = evictFlag;
    int32_t hitInclusive = hitFlag;
    for (uint32_t offset = 1; offset < LRU_FUSED_V3_WARP_SIZE;
         offset <<= 1) {
        int32_t priorEvict =
            asc_shfl_up(evictInclusive, offset, LRU_FUSED_V3_WARP_SIZE);
        int32_t priorHit =
            asc_shfl_up(hitInclusive, offset, LRU_FUSED_V3_WARP_SIZE);
        if (lane >= offset) {
            evictInclusive += priorEvict;
            hitInclusive += priorHit;
        }
    }
    if (lane == LRU_FUSED_V3_WARP_SIZE - 1U) {
        evictA[warp] = evictInclusive;
        hitA[warp] = hitInclusive;
    }
    asc_syncthreads();

    if (warp == 0) {
        int32_t warpEvict = evictA[lane];
        int32_t warpHit = hitA[lane];
        int32_t warpEvictInclusive = warpEvict;
        int32_t warpHitInclusive = warpHit;
        for (uint32_t offset = 1; offset < LRU_FUSED_V3_WARP_COUNT;
             offset <<= 1) {
            int32_t priorEvict = asc_shfl_up(
                warpEvictInclusive, offset, LRU_FUSED_V3_WARP_SIZE);
            int32_t priorHit = asc_shfl_up(
                warpHitInclusive, offset, LRU_FUSED_V3_WARP_SIZE);
            if (lane >= offset) {
                warpEvictInclusive += priorEvict;
                warpHitInclusive += priorHit;
            }
        }
        evictB[lane] = warpEvictInclusive - warpEvict;
        hitB[lane] = warpHitInclusive - warpHit;
    }
    asc_syncthreads();

    evictA[thread] = evictB[warp] + evictInclusive;
    hitA[thread] = hitB[warp] + hitInclusive;
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void StableScanPair1024(
    __ubuf__ int32_t *workspace, int32_t evictFlag, int32_t hitFlag,
    uint32_t thread)
{
    __ubuf__ int32_t *evictA = workspace + SCAN_EVICT_A_OFFSET;
    __ubuf__ int32_t *evictB = workspace + SCAN_EVICT_B_OFFSET;
    __ubuf__ int32_t *hitA = workspace + SCAN_HIT_A_OFFSET;
    __ubuf__ int32_t *hitB = workspace + SCAN_HIT_B_OFFSET;
    WarpScanPair1024(evictA, evictB, hitA, hitB, evictFlag, hitFlag,
                     thread);
}

__simt_callee__ __aicore__ inline void FusedV3ResetPrepareUb(
    __ubuf__ int32_t *workspace, __gm__ int64_t *reqId,
    __gm__ int64_t *lastReqId, __gm__ int32_t *stablePrefix)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *slotToToken = workspace + SLOT_TO_TOKEN_OFFSET;
    __ubuf__ int32_t *lruSlots = workspace + LRU_SLOTS_OFFSET;
    __ubuf__ int32_t *currentSlots = workspace + CURRENT_SLOTS_OFFSET;
    __ubuf__ int32_t *missTokens = workspace + MISS_TOKENS_OFFSET;
    __ubuf__ int32_t *missSlots = workspace + MISS_SLOTS_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;

    if (thread == 0) {
        control[CONTROL_CURRENT_EPOCH] = LRU_FUSED_V3_EPOCH;
        control[CONTROL_EVICTABLE_COUNT] = 0;
        control[CONTROL_HIT_COUNT] = 0;
        control[CONTROL_MISS_COUNT] = 0;
        control[CONTROL_RESET_ROW] = *lastReqId != *reqId ? 1 : 0;
        int32_t prefix = *stablePrefix;
        if (prefix < 0) {
            prefix = 0;
        } else if (prefix > static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN)) {
            prefix = static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN);
        }
        control[CONTROL_STABLE_PREFIX] = prefix;
    }
    asc_syncthreads();

    bool resetRow = control[CONTROL_RESET_ROW] != 0;
    int32_t stablePrefixLen = control[CONTROL_STABLE_PREFIX];
    for (uint32_t pos = thread; pos < LRU_FUSED_V3_TOPK;
         pos += LRU_FUSED_V3_THREADS) {
        currentSlots[pos] = LRU_INVALID_TOKEN;
        missTokens[pos] = LRU_INVALID_TOKEN;
        missSlots[pos] = LRU_INVALID_TOKEN;
    }
    for (uint32_t slot = thread; slot < LRU_FUSED_V3_CAPACITY;
         slot += LRU_FUSED_V3_THREADS) {
        if (resetRow) {
            slotToToken[slot] = LRU_INVALID_TOKEN;
            lruSlots[slot] = static_cast<int32_t>(slot);
        } else {
            int32_t token = slotToToken[slot];
            if (token >= 0 &&
                token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN) &&
                token >= stablePrefixLen) {
                slotToToken[slot] = LRU_INVALID_TOKEN;
            }
        }
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void FusedV3ResetFastPath(
    __ubuf__ int32_t *workspace)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *topkIndices = workspace + TOPK_OFFSET;
    __ubuf__ int32_t *slotToToken = workspace + SLOT_TO_TOKEN_OFFSET;
    __ubuf__ int32_t *lruSlots = workspace + LRU_SLOTS_OFFSET;
    __ubuf__ int32_t *currentSlots = workspace + CURRENT_SLOTS_OFFSET;
    __ubuf__ int32_t *missTokens = workspace + MISS_TOKENS_OFFSET;
    __ubuf__ int32_t *missSlots = workspace + MISS_SLOTS_OFFSET;
    __ubuf__ int32_t *scanResult = workspace + SCAN_EVICT_RESULT_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;

    int32_t missBase = 0;
    for (uint32_t tile = 0; tile < LRU_FUSED_V3_TOPK;
         tile += LRU_FUSED_V3_THREADS) {
        uint32_t pos = tile + thread;
        int32_t token = topkIndices[pos];
        int32_t validFlag = token >= 0 &&
            token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN);
        StableScanPair1024(workspace, validFlag, 0, thread);

        int32_t rank = missBase + scanResult[thread] - validFlag;
        if (validFlag != 0 &&
            rank < static_cast<int32_t>(LRU_FUSED_V3_CAPACITY)) {
            currentSlots[pos] = rank;
            missTokens[rank] = token;
            missSlots[rank] = rank;
            slotToToken[rank] = token;
        }
        missBase += scanResult[LRU_FUSED_V3_THREADS - 1];
        asc_syncthreads();
    }

    int32_t assignCount = missBase;
    if (assignCount > static_cast<int32_t>(LRU_FUSED_V3_CAPACITY)) {
        assignCount = static_cast<int32_t>(LRU_FUSED_V3_CAPACITY);
    }
    if (thread == 0) {
        control[CONTROL_MISS_COUNT] = assignCount;
    }
    int32_t remainingCount =
        static_cast<int32_t>(LRU_FUSED_V3_CAPACITY) - assignCount;
    for (uint32_t output = thread; output < LRU_FUSED_V3_CAPACITY;
         output += LRU_FUSED_V3_THREADS) {
        lruSlots[output] = static_cast<int32_t>(output) < remainingCount ?
            assignCount + static_cast<int32_t>(output) :
            static_cast<int32_t>(output) - remainingCount;
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void FusedV3AssignRebuildUb(
    __ubuf__ int32_t *workspace)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *slotToToken = workspace + SLOT_TO_TOKEN_OFFSET;
    __ubuf__ int32_t *lruSlots = workspace + LRU_SLOTS_OFFSET;
    __ubuf__ int32_t *currentSlots = workspace + CURRENT_SLOTS_OFFSET;
    __ubuf__ int32_t *missTokens = workspace + MISS_TOKENS_OFFSET;
    __ubuf__ int32_t *tokenMark = workspace + TOKEN_MARK_OFFSET;
    __ubuf__ int32_t *tokenPos = workspace + TOKEN_POS_OFFSET;
    __ubuf__ int32_t *missSlots = workspace + MISS_SLOTS_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;

    int32_t assignCount = control[CONTROL_MISS_COUNT];
    for (int32_t missIndex = static_cast<int32_t>(thread);
         missIndex < assignCount;
         missIndex += static_cast<int32_t>(LRU_FUSED_V3_THREADS)) {
        int32_t slot = tokenMark[missIndex];
        int32_t token = missTokens[missIndex];
        int32_t pos = tokenPos[missIndex];
        if (slot >= 0 && slot < static_cast<int32_t>(LRU_FUSED_V3_CAPACITY) &&
            pos >= 0 && pos < static_cast<int32_t>(LRU_FUSED_V3_TOPK)) {
            slotToToken[slot] = token;
            currentSlots[pos] = slot;
            missSlots[missIndex] = slot;
        }
    }
    asc_syncthreads();

    int32_t evictableCount = tokenPos[LRU_FUSED_V3_MAX_TOKEN - 2];
    int32_t hitCount = tokenPos[LRU_FUSED_V3_MAX_TOKEN - 1];
    int32_t unassignedCount = evictableCount - assignCount;
    int32_t validCount = evictableCount + hitCount;
    for (uint32_t output = thread; output < LRU_FUSED_V3_CAPACITY;
         output += LRU_FUSED_V3_THREADS) {
        int32_t slot = LRU_INVALID_TOKEN;
        if (static_cast<int32_t>(output) < unassignedCount) {
            slot = tokenMark[assignCount + output];
        } else if (static_cast<int32_t>(output) < evictableCount) {
            slot = missSlots[output - unassignedCount];
        } else if (static_cast<int32_t>(output) < validCount) {
            int32_t hitIndex = static_cast<int32_t>(output) - evictableCount;
            slot = tokenMark[LRU_FUSED_V3_MAX_TOKEN - 1 - hitIndex];
        }
        lruSlots[output] = slot;
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void PublishFusedV3StateDirectGm(
    __ubuf__ int32_t *workspace, __gm__ int64_t *reqId,
    __gm__ int64_t *lastReqIdOut, __gm__ int32_t *slotToTokenOut,
    __gm__ int32_t *lruSlotsOut, __gm__ int32_t *currentSlotsOut,
    __gm__ int32_t *missCountOut, __gm__ int32_t *missTokensOut,
    __gm__ int32_t *missSlotsOut, __gm__ int32_t *epochOut)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *slotToToken = workspace + SLOT_TO_TOKEN_OFFSET;
    __ubuf__ int32_t *lruSlots = workspace + LRU_SLOTS_OFFSET;
    __ubuf__ int32_t *currentSlots = workspace + CURRENT_SLOTS_OFFSET;
    __ubuf__ int32_t *missTokens = workspace + MISS_TOKENS_OFFSET;
    __ubuf__ int32_t *missSlots = workspace + MISS_SLOTS_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;

    for (uint32_t index = thread; index < LRU_FUSED_V3_CAPACITY;
         index += LRU_FUSED_V3_THREADS) {
        slotToTokenOut[index] = slotToToken[index];
        lruSlotsOut[index] = lruSlots[index];
    }
    for (uint32_t index = thread; index < LRU_FUSED_V3_TOPK;
         index += LRU_FUSED_V3_THREADS) {
        currentSlotsOut[index] = currentSlots[index];
        missTokensOut[index] = missTokens[index];
        missSlotsOut[index] = missSlots[index];
    }
    if (thread == 0) {
        *lastReqIdOut = *reqId;
        *missCountOut = control[CONTROL_MISS_COUNT];
        // V3 rebuilds token workspace every invocation, so epoch history no
        // longer participates in correctness. Keep the tested ABI state.
        *epochOut = LRU_FUSED_V3_EPOCH;
    }
}

__simt_callee__ __aicore__ inline void RecordV3Stage(
    __gm__ uint64_t *profileOut, uint32_t profileEnabled, uint32_t stage)
{
    if (profileEnabled != 0 && threadIdx.x == 0 &&
        stage < V3_PROFILE_STAMPS) {
        profileOut[stage] = clock();
    }
}

// Reduce a per-thread predicate to one block-wide value without a full 1024
// lane prefix.  Five warp shuffles plus two block barriers are sufficient.
__simt_callee__ __aicore__ inline int32_t BlockAny1024(
    __ubuf__ int32_t *workspace, int32_t predicate, uint32_t thread)
{
    uint32_t lane = thread & (LRU_FUSED_V3_WARP_SIZE - 1U);
    uint32_t warp = thread / LRU_FUSED_V3_WARP_SIZE;
    int32_t any = predicate;
    for (uint32_t offset = 1; offset < LRU_FUSED_V3_WARP_SIZE;
         offset <<= 1) {
        int32_t prior = asc_shfl_up(any, offset, LRU_FUSED_V3_WARP_SIZE);
        if (lane >= offset && prior != 0) {
            any = 1;
        }
    }
    __ubuf__ int32_t *warpAny = workspace + SCAN_EVICT_A_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;
    if (lane == LRU_FUSED_V3_WARP_SIZE - 1U) {
        warpAny[warp] = any;
    }
    asc_syncthreads();
    if (warp == 0) {
        int32_t blockAny = warpAny[lane];
        for (uint32_t offset = 1; offset < LRU_FUSED_V3_WARP_COUNT;
             offset <<= 1) {
            int32_t prior =
                asc_shfl_up(blockAny, offset, LRU_FUSED_V3_WARP_SIZE);
            if (lane >= offset && prior != 0) {
                blockAny = 1;
            }
        }
        if (lane == LRU_FUSED_V3_WARP_SIZE - 1U) {
            control[CONTROL_HAS_MISS] = blockAny;
        }
    }
    asc_syncthreads();
    return control[CONTROL_HAS_MISS];
}

// V3 replaces two serial/quadratic ownership operations with associative UB
// atomics. atomicMin is exactly Top-K first occurrence; atomicMax is exactly
// old-LRU global last-wins. Neither result depends on lane scheduling.
__simt_callee__ __aicore__ inline void FusedV3NormalPlan(
    __ubuf__ int32_t *workspace, __gm__ uint64_t *profileOut,
    uint32_t profileEnabled)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *topkIndices = workspace + TOPK_OFFSET;
    __ubuf__ int32_t *slotToToken = workspace + SLOT_TO_TOKEN_OFFSET;
    __ubuf__ int32_t *lruSlots = workspace + LRU_SLOTS_OFFSET;
    __ubuf__ int32_t *currentSlots = workspace + CURRENT_SLOTS_OFFSET;
    __ubuf__ int32_t *missTokens = workspace + MISS_TOKENS_OFFSET;
    __ubuf__ int32_t *tokenMark = workspace + TOKEN_MARK_OFFSET;
    __ubuf__ int32_t *tokenPos = workspace + TOKEN_POS_OFFSET;
    __ubuf__ int32_t *scanEvictResult =
        workspace + SCAN_EVICT_RESULT_OFFSET;
    __ubuf__ int32_t *scanHitResult = workspace + SCAN_HIT_RESULT_OFFSET;
    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;

    for (uint32_t token = thread; token < LRU_FUSED_V3_MAX_TOKEN;
         token += LRU_FUSED_V3_THREADS) {
        tokenMark[token] = LRU_INVALID_TOKEN;
        tokenPos[token] = static_cast<int32_t>(LRU_FUSED_V3_TOPK);
    }
    asc_syncthreads();

    for (uint32_t pos = thread; pos < LRU_FUSED_V3_TOPK;
         pos += LRU_FUSED_V3_THREADS) {
        int32_t token = topkIndices[pos];
        if (token >= 0 &&
            token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN)) {
            asc_atomic_min(tokenPos + token, static_cast<int32_t>(pos));
        }
    }
    asc_syncthreads();
    RecordV3Stage(profileOut, profileEnabled, 2);

    // Encode all old-LRU entries and concurrently elect the greatest global
    // old-LRU order for every resident token. This removes V3's O(tile^2)
    // later-lane search, which is largest in the all-hit workload.
    for (uint32_t order = thread; order < LRU_FUSED_V3_CAPACITY;
         order += LRU_FUSED_V3_THREADS) {
        int32_t slot = lruSlots[order];
        int32_t encoded = LRU_INVALID_TOKEN;
        if (slot >= 0 && slot < static_cast<int32_t>(LRU_FUSED_V3_CAPACITY)) {
            int32_t token = slotToToken[slot];
            if (token >= 0 &&
                token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN) &&
                tokenPos[token] < static_cast<int32_t>(LRU_FUSED_V3_TOPK)) {
                encoded = -slot - 2;
                asc_atomic_max(tokenMark + token,
                               static_cast<int32_t>(order));
            } else {
                encoded = slot;
            }
        }
        lruSlots[order] = encoded;
    }
    asc_syncthreads();

    for (uint32_t order = thread; order < LRU_FUSED_V3_CAPACITY;
         order += LRU_FUSED_V3_THREADS) {
        int32_t encoded = lruSlots[order];
        if (encoded <= -2) {
            int32_t slot = -encoded - 2;
            int32_t token = slotToToken[slot];
            if (tokenMark[token] == static_cast<int32_t>(order)) {
                currentSlots[tokenPos[token]] = slot;
            }
        }
    }
    asc_syncthreads();
    RecordV3Stage(profileOut, profileEnabled, 3);

    int32_t evictBase = 0;
    int32_t hitBase = 0;
    for (uint32_t tile = 0; tile < LRU_FUSED_V3_CAPACITY;
         tile += LRU_FUSED_V3_THREADS) {
        uint32_t order = tile + thread;
        int32_t encoded = lruSlots[order];
        int32_t evictFlag = encoded >= 0 &&
            encoded < static_cast<int32_t>(LRU_FUSED_V3_CAPACITY);
        int32_t hitFlag = encoded <= -2;
        StableScanPair1024(workspace, evictFlag, hitFlag, thread);
        int32_t evictRank = scanEvictResult[thread] - evictFlag;
        int32_t hitRank = scanHitResult[thread] - hitFlag;
        if (evictFlag != 0) {
            tokenMark[evictBase + evictRank] = encoded;
        }
        if (hitFlag != 0) {
            int32_t output = static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN) -
                             1 - hitBase - hitRank;
            tokenMark[output] = -encoded - 2;
        }
        evictBase += scanEvictResult[LRU_FUSED_V3_THREADS - 1];
        hitBase += scanHitResult[LRU_FUSED_V3_THREADS - 1];
        asc_syncthreads();
    }
    RecordV3Stage(profileOut, profileEnabled, 4);

    int32_t localHasMiss = 0;
    for (uint32_t pos = thread; pos < LRU_FUSED_V3_TOPK;
         pos += LRU_FUSED_V3_THREADS) {
        int32_t token = topkIndices[pos];
        if (token >= 0 &&
            token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN) &&
            currentSlots[pos] < 0) {
            localHasMiss = 1;
        }
    }
    int32_t hasMiss = BlockAny1024(workspace, localHasMiss, thread);

    int32_t missBase = 0;
    if (hasMiss != 0) {
        for (uint32_t tile = 0; tile < LRU_FUSED_V3_TOPK;
             tile += LRU_FUSED_V3_THREADS) {
            uint32_t pos = tile + thread;
            int32_t token = topkIndices[pos];
            int32_t missFlag = token >= 0 &&
                token < static_cast<int32_t>(LRU_FUSED_V3_MAX_TOKEN) &&
                currentSlots[pos] < 0;
            StableScanPair1024(workspace, missFlag, 0, thread);
            int32_t missRank = scanEvictResult[thread] - missFlag;
            if (missFlag != 0) {
                int32_t output = missBase + missRank;
                missTokens[output] = token;
                tokenPos[output] = static_cast<int32_t>(pos);
            }
            missBase += scanEvictResult[LRU_FUSED_V3_THREADS - 1];
            asc_syncthreads();
        }
    }

    if (thread == 0) {
        control[CONTROL_EVICTABLE_COUNT] = evictBase;
        control[CONTROL_HIT_COUNT] = hitBase;
        control[CONTROL_MISS_COUNT] = missBase < evictBase ?
            missBase : evictBase;
        tokenPos[LRU_FUSED_V3_MAX_TOKEN - 2] = evictBase;
        tokenPos[LRU_FUSED_V3_MAX_TOKEN - 1] = hitBase;
    }
    asc_syncthreads();
    RecordV3Stage(profileOut, profileEnabled, 5);
}

// One VF owns the complete Compact operation. Intermediate state remains in
// the same UB workspace; only final externally visible state is written to GM.
__simt_callee__ __aicore__ inline void LruCompactFusedV3Body(
    __ubuf__ int32_t *workspace, __gm__ int64_t *reqId,
    __gm__ int64_t *lastReqIdOut, __gm__ int32_t *stablePrefix,
    __gm__ int32_t *slotToTokenOut, __gm__ int32_t *lruSlotsOut,
    __gm__ int32_t *currentSlotsOut, __gm__ int32_t *missCountOut,
    __gm__ int32_t *missTokensOut, __gm__ int32_t *missSlotsOut,
    __gm__ int32_t *epochOut, __gm__ uint64_t *profileOut,
    uint32_t profileEnabled)
{
    RecordV3Stage(profileOut, profileEnabled, 0);
    FusedV3ResetPrepareUb(workspace, reqId, lastReqIdOut, stablePrefix);
    RecordV3Stage(profileOut, profileEnabled, 1);

    __ubuf__ int32_t *control = workspace + SCAN_CONTROL_OFFSET;
    if (control[CONTROL_RESET_ROW] != 0) {
        FusedV3ResetFastPath(workspace);
    } else {
        FusedV3NormalPlan(workspace, profileOut, profileEnabled);
        FusedV3AssignRebuildUb(workspace);
    }

    RecordV3Stage(profileOut, profileEnabled, 6);
    PublishFusedV3StateDirectGm(
        workspace, reqId, lastReqIdOut, slotToTokenOut, lruSlotsOut,
        currentSlotsOut, missCountOut, missTokensOut, missSlotsOut,
        epochOut);
    RecordV3Stage(profileOut, profileEnabled, 7);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(LRU_FUSED_V3_THREADS) inline void
LruCompactFusedV3Vf(
    __ubuf__ int32_t *workspace, __gm__ int64_t *reqId,
    __gm__ int64_t *lastReqIdOut, __gm__ int32_t *stablePrefix,
    __gm__ int32_t *slotToTokenOut, __gm__ int32_t *lruSlotsOut,
    __gm__ int32_t *currentSlotsOut, __gm__ int32_t *missCountOut,
    __gm__ int32_t *missTokensOut, __gm__ int32_t *missSlotsOut,
    __gm__ int32_t *epochOut, __gm__ uint64_t *profileOut,
    uint32_t profileEnabled)
{
    LruCompactFusedV3Body(
        workspace, reqId, lastReqIdOut, stablePrefix, slotToTokenOut,
        lruSlotsOut, currentSlotsOut, missCountOut, missTokensOut,
        missSlotsOut, epochOut, profileOut, profileEnabled);
}

__aicore__ inline void CopyGmToUb(
    __ubuf__ int32_t *destination, __gm__ int32_t *source,
    uint32_t blocks)
{
    copy_gm_to_ubuf(destination, source, 0, 1, blocks, 0, 0);
}

} // namespace

extern "C" __global__ __vector__ void OffloadLruCompactFusedV3Kernel(
    GM_ADDR reqIds, GM_ADDR lastReqIds, GM_ADDR topkIndices,
    GM_ADDR stablePrefixLens, GM_ADDR slotToToken, GM_ADDR lruSlots,
    GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens,
    GM_ADDR missSlots, GM_ADDR epoch, GM_ADDR profileOutput, int64_t row,
    uint32_t profileEnabled)
{
    AscendC::InitSocState();
    constexpr event_t MTE2_V_EVENT = static_cast<event_t>(0);
    extern __ubuf__ int32_t workspace[];

    auto *reqIdRow = reinterpret_cast<__gm__ int64_t *>(reqIds) + row;
    auto *lastReqIdRow = reinterpret_cast<__gm__ int64_t *>(lastReqIds) + row;
    auto *topkIndicesRow = reinterpret_cast<__gm__ int32_t *>(topkIndices) +
                           row * LRU_FUSED_V3_TOPK;
    auto *stablePrefixRow =
        reinterpret_cast<__gm__ int32_t *>(stablePrefixLens) + row;
    auto *slotToTokenRow = reinterpret_cast<__gm__ int32_t *>(slotToToken) +
                           row * LRU_FUSED_V3_CAPACITY;
    auto *lruSlotsRow = reinterpret_cast<__gm__ int32_t *>(lruSlots) +
                        row * LRU_FUSED_V3_CAPACITY;
    auto *currentSlotsRow = reinterpret_cast<__gm__ int32_t *>(currentSlots) +
                            row * LRU_FUSED_V3_TOPK;
    auto *missCountRow = reinterpret_cast<__gm__ int32_t *>(missCount) + row;
    auto *missTokensRow = reinterpret_cast<__gm__ int32_t *>(missTokens) +
                          row * LRU_FUSED_V3_TOPK;
    auto *missSlotsRow = reinterpret_cast<__gm__ int32_t *>(missSlots) +
                         row * LRU_FUSED_V3_TOPK;
    auto *epochRow = reinterpret_cast<__gm__ int32_t *>(epoch);
    auto *profileRow = reinterpret_cast<__gm__ uint64_t *>(profileOutput);

    // Only true inputs are staged: TopK, slot_to_token and the old LRU row.
    // current/miss arrays are initialized in UB, and token workspace is local.
    CopyGmToUb(workspace + TOPK_OFFSET, topkIndicesRow,
               LRU_FUSED_V3_TOPK_BLOCKS);
    CopyGmToUb(workspace + SLOT_TO_TOKEN_OFFSET, slotToTokenRow,
               LRU_FUSED_V3_CAPACITY_BLOCKS);
    CopyGmToUb(workspace + LRU_SLOTS_OFFSET, lruSlotsRow,
               LRU_FUSED_V3_CAPACITY_BLOCKS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(MTE2_V_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(MTE2_V_EVENT);

    asc_vf_call<LruCompactFusedV3Vf>(
        dim3(LRU_FUSED_V3_THREADS), workspace, reqIdRow, lastReqIdRow,
        stablePrefixRow, slotToTokenRow, lruSlotsRow, currentSlotsRow,
        missCountRow, missTokensRow, missSlotsRow, epochRow, profileRow,
        profileEnabled);
}

namespace {

void LaunchFusedV3(
    uint64_t reqIds, uint64_t lastReqIds, uint64_t topkIndices,
    uint64_t stablePrefixLens, uint64_t slotToToken, uint64_t lruSlots,
    uint64_t currentSlots, uint64_t missCount, uint64_t missTokens,
    uint64_t missSlots, uint64_t tokenMarkWorkspace,
    uint64_t tokenPosWorkspace, uint64_t epochs, int64_t row,
    int64_t topk, int64_t capacity, int64_t maxToken, void *stream,
    uint32_t profileEnabled)
{
    if (row < 0 || topk != LRU_FUSED_V3_TOPK ||
        capacity != LRU_FUSED_V3_CAPACITY ||
        maxToken != LRU_FUSED_V3_MAX_TOKEN) {
        return;
    }

    // These ABI workspaces are no longer part of Compact state. token_mark is
    // reused only as an optional profile-output area; token_pos is untouched.
    (void)tokenPosWorkspace;
    int64_t workspaceIndex = row % LRU_FUSED_V3_WORKSPACE_THREADS;
    auto *epochRow = reinterpret_cast<int32_t *>(epochs) + workspaceIndex;
    auto *epochBytes = reinterpret_cast<uint8_t *>(epochRow);
    auto *profileRow = reinterpret_cast<uint64_t *>(tokenMarkWorkspace) +
                       workspaceIndex * (LRU_FUSED_V3_MAX_TOKEN / 2U);
    auto *profileBytes = reinterpret_cast<uint8_t *>(profileRow);

    OffloadLruCompactFusedV3Kernel<<<1, FUSED_V3_DYNAMIC_UB_BYTES, stream>>>(
        reinterpret_cast<GM_ADDR>(reqIds),
        reinterpret_cast<GM_ADDR>(lastReqIds),
        reinterpret_cast<GM_ADDR>(topkIndices),
        reinterpret_cast<GM_ADDR>(stablePrefixLens),
        reinterpret_cast<GM_ADDR>(slotToToken),
        reinterpret_cast<GM_ADDR>(lruSlots),
        reinterpret_cast<GM_ADDR>(currentSlots),
        reinterpret_cast<GM_ADDR>(missCount),
        reinterpret_cast<GM_ADDR>(missTokens),
        reinterpret_cast<GM_ADDR>(missSlots), epochBytes, profileBytes, row,
        profileEnabled);
}

} // namespace

extern "C" void OffloadOpsLruCompactFusedV3(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
    uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
    uint64_t miss_slots, uint64_t token_mark_workspace,
    uint64_t token_pos_workspace, uint64_t epochs, int64_t row,
    int64_t topk, int64_t capacity, int64_t max_token, void *stream,
    uint32_t profile_enabled)
{
    LaunchFusedV3(
        req_ids, last_req_ids, topk_indices, stable_prefix_lens,
        slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
        miss_slots, token_mark_workspace, token_pos_workspace, epochs, row,
        topk, capacity, max_token, stream, profile_enabled);
}
