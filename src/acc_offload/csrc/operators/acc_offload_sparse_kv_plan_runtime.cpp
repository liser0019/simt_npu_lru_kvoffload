/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Runtime-shape Sparse KV planning kernel for Ascend 950PR/950DT.
 *
 * Unlike Compact V3, this implementation never allocates token-domain-sized
 * state.  A per-request open-addressed hash is sized from topk, while stable
 * compaction scratch is sized from topk/capacity.  max_token is only a token
 * validity bound.  One mixed-kernel launch schedules request rows over up to
 * 32 Vector cores; each row is processed by a 1024-lane SIMT VF.
 */

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

#include "acc_offload_sparse_kv_plan_runtime.h"
#include "acc_offload_sparse_kv_runtime.h"

namespace {

constexpr uint32_t PLAN_THREADS = 1024;
constexpr uint32_t PLAN_WARP_SIZE = 32;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_MAX_BLOCKS = 32;
constexpr int32_t INVALID_TOKEN = -1;
constexpr int32_t HASH_EMPTY = -1;
constexpr int32_t HASH_POSITION_EMPTY = INT32_MAX;
constexpr uint32_t HASH_BUCKET_INVALID = UINT32_MAX;

constexpr uint32_t WARP_TOTALS_OFFSET = 0;
constexpr uint32_t WARP_BASES_OFFSET =
    WARP_TOTALS_OFFSET + PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_OFFSET =
    WARP_BASES_OFFSET + PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_TOTAL = 0;
constexpr uint32_t CONTROL_RESET = 1;
constexpr uint32_t CONTROL_ELEMENTS = 8;
constexpr uint32_t PLAN_DYNAMIC_UB_ELEMENTS =
    CONTROL_OFFSET + CONTROL_ELEMENTS;
constexpr uint32_t PLAN_DYNAMIC_UB_BYTES =
    PLAN_DYNAMIC_UB_ELEMENTS * sizeof(int32_t);

static_assert(PLAN_WARP_COUNT == 32U,
              "1024 lanes must form 32 Ascend warps");
static_assert(PLAN_DYNAMIC_UB_BYTES == 288U,
              "runtime Plan scan scratch must remain 288 bytes");
static_assert(PLAN_DYNAMIC_UB_BYTES % 32U == 0U,
              "dynamic UB must be 32-byte aligned");

__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan1024(
    __ubuf__ int32_t *workspace, int32_t flag, uint32_t thread)
{
    __ubuf__ int32_t *warpTotals = workspace + WARP_TOTALS_OFFSET;
    __ubuf__ int32_t *warpBases = workspace + WARP_BASES_OFFSET;
    __ubuf__ int32_t *control = workspace + CONTROL_OFFSET;
    uint32_t lane = thread & (PLAN_WARP_SIZE - 1U);
    uint32_t warp = thread / PLAN_WARP_SIZE;

    int32_t inclusive = flag;
    for (uint32_t offset = 1; offset < PLAN_WARP_SIZE; offset <<= 1U) {
        int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
        if (lane >= offset) {
            inclusive += prior;
        }
    }
    if (lane == PLAN_WARP_SIZE - 1U) {
        warpTotals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t value = warpTotals[lane];
        int32_t warpInclusive = value;
        for (uint32_t offset = 1; offset < PLAN_WARP_COUNT;
             offset <<= 1U) {
            int32_t prior =
                asc_shfl_up(warpInclusive, offset, PLAN_WARP_SIZE);
            if (lane >= offset) {
                warpInclusive += prior;
            }
        }
        warpBases[lane] = warpInclusive - value;
    }
    asc_syncthreads();

    int32_t exclusive = warpBases[warp] + inclusive - flag;
    if (thread == 0U) {
        control[CONTROL_TOTAL] =
            warpBases[PLAN_WARP_COUNT - 1U] +
            warpTotals[PLAN_WARP_COUNT - 1U];
    }
    asc_syncthreads();
    return exclusive;
}

__simt_callee__ __aicore__ inline uint32_t HashToken(
    int32_t token, uint64_t hashCapacity)
{
    uint32_t value = static_cast<uint32_t>(token);
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    // The int32 TopK-position ABI limits hashCapacity to at most 2^31.
    // Bucket indices remain uint32 so the legal upper half of that range can
    // never alias a negative signed sentinel.
    return value & static_cast<uint32_t>(hashCapacity - 1U);
}

// Insert returns the stable bucket selected by linear probing.  The key is
// published with GM CAS; all colliding copies of the same TopK token then use
// atomicMin to reproduce the CPU oracle's first-occurrence position.
__simt_callee__ __aicore__ inline uint32_t InsertTopkToken(
    __gm__ int32_t *keys, __gm__ int32_t *firstPositions,
    int32_t token, int32_t position, uint64_t hashCapacity)
{
    uint32_t bucket = HashToken(token, hashCapacity);
    uint32_t mask = static_cast<uint32_t>(hashCapacity - 1U);
    for (uint64_t probe = 0; probe < hashCapacity; ++probe) {
        int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
        if (old == HASH_EMPTY || old == token) {
            asc_atomic_min(firstPositions + bucket, position);
            return bucket;
        }
        bucket = (bucket + 1U) & mask;
    }
    return HASH_BUCKET_INVALID;
}

__simt_callee__ __aicore__ inline uint32_t LookupTopkToken(
    __gm__ int32_t *keys, int32_t token, uint64_t hashCapacity)
{
    uint32_t bucket = HashToken(token, hashCapacity);
    uint32_t mask = static_cast<uint32_t>(hashCapacity - 1U);
    for (uint64_t probe = 0; probe < hashCapacity; ++probe) {
        int32_t key = keys[bucket];
        if (key == token) {
            return bucket;
        }
        if (key == HASH_EMPTY) {
            return HASH_BUCKET_INVALID;
        }
        bucket = (bucket + 1U) & mask;
    }
    return HASH_BUCKET_INVALID;
}

__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t *scanWorkspace, int64_t row,
    __gm__ int64_t *reqIds, __gm__ int64_t *lastReqIds,
    __gm__ int32_t *topkIndices, __gm__ int32_t *stablePrefixLens,
    __gm__ int32_t *slotToToken, __gm__ int32_t *lruSlots,
    __gm__ int32_t *currentSlots, __gm__ int32_t *missCount,
    __gm__ int32_t *missTokens, __gm__ int32_t *missSlots,
    __gm__ int32_t *compactWorkspace, uint64_t hashCapacity,
    uint64_t workspaceRowElements, int64_t topk, int64_t capacity,
    int64_t maxToken)
{
    uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ int32_t *control = scanWorkspace + CONTROL_OFFSET;
    int64_t topkBase = row * topk;
    int64_t capacityBase = row * capacity;

    __gm__ int32_t *rowWorkspace =
        compactWorkspace + row * workspaceRowElements;
    __gm__ int32_t *hashKeys = rowWorkspace;
    __gm__ int32_t *hashFirstPos = hashKeys + hashCapacity;
    __gm__ int32_t *hashOwner = hashFirstPos + hashCapacity;
    __gm__ int32_t *evictableSlots = hashOwner + hashCapacity;
    __gm__ int32_t *hitSlots = evictableSlots + capacity;
    __gm__ int32_t *missPositions = hitSlots + capacity;

    // All 1024 lanes participate in every barrier.  Runtime tails contribute
    // no work; they never return early from the VF.
    for (int64_t pos = thread; pos < topk; pos += PLAN_THREADS) {
        currentSlots[topkBase + pos] = INVALID_TOKEN;
        missTokens[topkBase + pos] = INVALID_TOKEN;
        missSlots[topkBase + pos] = INVALID_TOKEN;
    }
    for (uint64_t index = thread; index < hashCapacity;
         index += PLAN_THREADS) {
        hashKeys[index] = HASH_EMPTY;
        hashFirstPos[index] = HASH_POSITION_EMPTY;
        hashOwner[index] = INVALID_TOKEN;
    }
    if (thread == 0U) {
        missCount[row] = 0;
        control[CONTROL_RESET] = lastReqIds[row] != reqIds[row] ? 1 : 0;
    }
    asc_syncthreads();

    if (control[CONTROL_RESET] != 0) {
        for (int64_t slot = thread; slot < capacity;
             slot += PLAN_THREADS) {
            slotToToken[capacityBase + slot] = INVALID_TOKEN;
            lruSlots[capacityBase + slot] = static_cast<int32_t>(slot);
        }
        if (thread == 0U) {
            lastReqIds[row] = reqIds[row];
        }
    }
    asc_syncthreads();

    // Build an O(topk) active-token hash.  The load factor is <= 0.5, so a
    // valid insertion always has an empty bucket without a shape specialization.
    for (int64_t pos = thread; pos < topk; pos += PLAN_THREADS) {
        int32_t token = topkIndices[topkBase + pos];
        if (token >= 0 && static_cast<int64_t>(token) < maxToken) {
            (void)InsertTopkToken(
                hashKeys, hashFirstPos, token, static_cast<int32_t>(pos),
                hashCapacity);
        }
    }
    asc_syncthreads();

    int32_t stablePrefix = stablePrefixLens[row];
    if (stablePrefix < 0) {
        stablePrefix = 0;
    } else if (static_cast<int64_t>(stablePrefix) > maxToken) {
        stablePrefix = static_cast<int32_t>(maxToken);
    }

    int64_t evictableCount = 0;
    int64_t hitCount = 0;
    for (int64_t tile = 0; tile < capacity; tile += PLAN_THREADS) {
        int64_t order = tile + thread;
        int32_t slot = INVALID_TOKEN;
        int32_t hitFlag = 0;
        int32_t evictFlag = 0;
        uint32_t bucket = HASH_BUCKET_INVALID;
        if (order < capacity) {
            slot = lruSlots[capacityBase + order];
            if (slot >= 0 && static_cast<int64_t>(slot) < capacity) {
                int32_t token = slotToToken[capacityBase + slot];
                if (token >= 0 && static_cast<int64_t>(token) < maxToken &&
                    token >= stablePrefix) {
                    slotToToken[capacityBase + slot] = INVALID_TOKEN;
                    token = INVALID_TOKEN;
                }
                if (token >= 0 && static_cast<int64_t>(token) < maxToken) {
                    bucket = LookupTopkToken(hashKeys, token, hashCapacity);
                }
                if (bucket != HASH_BUCKET_INVALID) {
                    hitFlag = 1;
                    asc_atomic_max(hashOwner + bucket,
                                   static_cast<int32_t>(order));
                } else {
                    evictFlag = 1;
                }
            }
        }

        int32_t evictRank =
            BlockExclusiveScan1024(scanWorkspace, evictFlag, thread);
        int32_t tileEvictCount = control[CONTROL_TOTAL];
        int32_t hitRank =
            BlockExclusiveScan1024(scanWorkspace, hitFlag, thread);
        int32_t tileHitCount = control[CONTROL_TOTAL];
        if (evictFlag != 0) {
            evictableSlots[evictableCount + evictRank] = slot;
        }
        if (hitFlag != 0) {
            hitSlots[hitCount + hitRank] = slot;
        }
        asc_syncthreads();
        evictableCount += tileEvictCount;
        hitCount += tileHitCount;
    }

    // The CPU oracle overwrites current_slots[first_topk_position] for every
    // resident duplicate in old-LRU order.  atomicMax elected the same final
    // (greatest-order) owner deterministically.
    for (int64_t order = thread; order < capacity; order += PLAN_THREADS) {
        int32_t slot = lruSlots[capacityBase + order];
        if (slot < 0 || static_cast<int64_t>(slot) >= capacity) {
            continue;
        }
        int32_t token = slotToToken[capacityBase + slot];
        if (token < 0 || static_cast<int64_t>(token) >= maxToken) {
            continue;
        }
        uint32_t bucket = LookupTopkToken(hashKeys, token, hashCapacity);
        if (bucket != HASH_BUCKET_INVALID && hashOwner[bucket] == order) {
            int32_t position = hashFirstPos[bucket];
            if (position >= 0 && static_cast<int64_t>(position) < topk) {
                currentSlots[topkBase + position] = slot;
            }
        }
    }
    asc_syncthreads();

    int64_t localMissCount = 0;
    for (int64_t tile = 0; tile < topk; tile += PLAN_THREADS) {
        int64_t position = tile + thread;
        int32_t token = INVALID_TOKEN;
        int32_t missFlag = 0;
        if (position < topk) {
            token = topkIndices[topkBase + position];
            missFlag = token >= 0 && static_cast<int64_t>(token) < maxToken &&
                       currentSlots[topkBase + position] < 0;
        }
        int32_t missRank =
            BlockExclusiveScan1024(scanWorkspace, missFlag, thread);
        int32_t tileMissCount = control[CONTROL_TOTAL];
        if (missFlag != 0) {
            int64_t output = localMissCount + missRank;
            missPositions[output] = static_cast<int32_t>(position);
        }
        asc_syncthreads();
        localMissCount += tileMissCount;
    }

    int64_t assignCount =
        localMissCount < evictableCount ? localMissCount : evictableCount;
    for (int64_t miss = thread; miss < assignCount; miss += PLAN_THREADS) {
        int32_t slot = evictableSlots[miss];
        int32_t position = missPositions[miss];
        int32_t token = topkIndices[topkBase + position];
        slotToToken[capacityBase + slot] = token;
        currentSlots[topkBase + position] = slot;
        // Only misses that actually receive a resident slot are externally
        // visible.  The row was initialized to INVALID_TOKEN above, so the
        // unassigned tail remains identical to the CPU oracle contract.
        missTokens[topkBase + miss] = token;
        missSlots[topkBase + miss] = slot;
    }
    asc_syncthreads();

    int64_t unassignedCount = evictableCount - assignCount;
    int64_t validLruCount = evictableCount + hitCount;
    for (int64_t output = thread; output < validLruCount;
         output += PLAN_THREADS) {
        int32_t slot = INVALID_TOKEN;
        if (output < unassignedCount) {
            slot = evictableSlots[assignCount + output];
        } else if (output < evictableCount) {
            slot = missSlots[topkBase + output - unassignedCount];
        } else {
            slot = hitSlots[output - evictableCount];
        }
        lruSlots[capacityBase + output] = slot;
    }
    if (thread == 0U) {
        missCount[row] = static_cast<int32_t>(assignCount);
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(PLAN_THREADS) inline void
SparseKvPlanRuntimeVf(
    __ubuf__ int32_t *scanWorkspace, __gm__ int64_t *reqIds,
    __gm__ int64_t *lastReqIds, __gm__ int32_t *topkIndices,
    __gm__ int32_t *stablePrefixLens, __gm__ int32_t *slotToToken,
    __gm__ int32_t *lruSlots, __gm__ int32_t *currentSlots,
    __gm__ int32_t *missCount, __gm__ int32_t *missTokens,
    __gm__ int32_t *missSlots, __gm__ int32_t *compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements,
    int64_t rowStart, int64_t rowStride, int64_t numReqs, int64_t topk,
    int64_t capacity, int64_t maxToken)
{
    for (int64_t row = rowStart; row < numReqs; row += rowStride) {
        ProcessRuntimeRow(
            scanWorkspace, row, reqIds, lastReqIds, topkIndices,
            stablePrefixLens, slotToToken, lruSlots, currentSlots, missCount,
            missTokens, missSlots, compactWorkspace, hashCapacity,
            workspaceRowElements, topk, capacity, maxToken);
    }
}

} // namespace

extern "C" __global__ __vector__ void SparseKvPlanRuntimeKernel(
    GM_ADDR reqIds, GM_ADDR lastReqIds, GM_ADDR topkIndices,
    GM_ADDR stablePrefixLens, GM_ADDR slotToToken, GM_ADDR lruSlots,
    GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens,
    GM_ADDR missSlots, GM_ADDR compactWorkspace, uint64_t hashCapacity,
    uint64_t workspaceRowElements, int64_t numReqs, int64_t topk,
    int64_t capacity, int64_t maxToken)
{
    AscendC::InitSocState();
    extern __ubuf__ int32_t scanWorkspace[];
    int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t rowStride = static_cast<int64_t>(AscendC::GetBlockNum());
    asc_vf_call<SparseKvPlanRuntimeVf>(
        dim3(PLAN_THREADS), scanWorkspace,
        reinterpret_cast<__gm__ int64_t *>(reqIds),
        reinterpret_cast<__gm__ int64_t *>(lastReqIds),
        reinterpret_cast<__gm__ int32_t *>(topkIndices),
        reinterpret_cast<__gm__ int32_t *>(stablePrefixLens),
        reinterpret_cast<__gm__ int32_t *>(slotToToken),
        reinterpret_cast<__gm__ int32_t *>(lruSlots),
        reinterpret_cast<__gm__ int32_t *>(currentSlots),
        reinterpret_cast<__gm__ int32_t *>(missCount),
        reinterpret_cast<__gm__ int32_t *>(missTokens),
        reinterpret_cast<__gm__ int32_t *>(missSlots),
        reinterpret_cast<__gm__ int32_t *>(compactWorkspace),
        hashCapacity, workspaceRowElements, rowStart, rowStride, numReqs,
        topk, capacity, maxToken);
}

void OffloadOpsSparseKvPlanRuntime(
    uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
    uint64_t stable_prefix_lens, uint64_t slot_to_token,
    uint64_t lru_slots, uint64_t current_slots, uint64_t miss_count,
    uint64_t miss_tokens, uint64_t miss_slots, uint64_t compact_workspace,
    uint64_t compact_workspace_bytes, int64_t num_reqs, int64_t topk,
    int64_t capacity, int64_t max_token, void *stream)
{
    uint64_t requiredBytes =
        sparse_kv_plan_workspace_size_bytes(num_reqs, topk, capacity);
    uint64_t hashCapacity = sparse_kv_plan_hash_capacity(topk);
    uint64_t rowElements =
        sparse_kv_plan_workspace_row_elements(topk, capacity);
    if (requiredBytes == 0 || compact_workspace_bytes < requiredBytes ||
        max_token <= 0 || max_token > INT32_MAX) {
        return;
    }
    uint32_t blockDim = num_reqs < PLAN_MAX_BLOCKS
        ? static_cast<uint32_t>(num_reqs)
        : PLAN_MAX_BLOCKS;
    SparseKvPlanRuntimeKernel<<<blockDim, PLAN_DYNAMIC_UB_BYTES, stream>>>(
        reinterpret_cast<GM_ADDR>(req_ids),
        reinterpret_cast<GM_ADDR>(last_req_ids),
        reinterpret_cast<GM_ADDR>(topk_indices),
        reinterpret_cast<GM_ADDR>(stable_prefix_lens),
        reinterpret_cast<GM_ADDR>(slot_to_token),
        reinterpret_cast<GM_ADDR>(lru_slots),
        reinterpret_cast<GM_ADDR>(current_slots),
        reinterpret_cast<GM_ADDR>(miss_count),
        reinterpret_cast<GM_ADDR>(miss_tokens),
        reinterpret_cast<GM_ADDR>(miss_slots),
        reinterpret_cast<GM_ADDR>(compact_workspace), hashCapacity,
        rowElements, num_reqs, topk, capacity, max_token);
}
