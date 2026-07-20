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
#ifndef __MEMFABRIC_ACC_OFFLOAD_H__
#define __MEMFABRIC_ACC_OFFLOAD_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OFFLOAD_SCENE_LOCAL = 0,                   /* single-card local DRAM memory pool */
    OFFLOAD_SCENE_SHARED = 1,                 /* multi-card shared DRAM memory pool */
} offload_scene_t;

typedef struct {
    uint32_t deviceId;                        /* Device ID to bind */
    uint64_t reserveSize;                     /* Reserved DRAM pool size in bytes, will be aligned up to GB. */
    uint64_t allocSize;                      /* Allocated local physical DRAM size in bytes, will be aligned
                                                 up to GB. LOCAL: must equal reserveSize; SHARED: provides
                                                 the actual size. */
    uint32_t worldSize;                       /* number of ranks in the group (used in SHARED scene) */
    uint32_t rankId;                          /* local rank id, 0 is the allocator (used in SHARED scene) */
    offload_scene_t scene;                    /* LOCAL: single-card pool; SHARED: multi-card shared pool */
} offload_config_t;

/**
 * @brief Initialize the offload module.
 *
 * This function initializes the hybm big memory entity and loads the
 * offload library for sparse copy operations.
 *
 * @param config  [in] Init config, see offload_config_t.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_init(const offload_config_t &config);

/**
 * @brief Uninitialize the offload module.
 *
 * Releases the hybm big memory entity, unloads the extend library and
 * performs cleanup. Safe to call when not initialized.
 */
void offload_uninit();

/**
 * @brief Allocate host memory from the offload memory pool.
 *
 * Allocates a contiguous block from the pre-reserved hybm host memory.
 * The returned pointer is 16-byte aligned.
 *
 * @param size  [in] Memory size in bytes.
 * @param flags [in] optional flags
 * @return Non-zero address on success, 0 on failure.
 */
uint64_t offload_malloc(uint64_t size, uint64_t flags);

/**
 * @brief Free host memory previously allocated by offload malloc.
 *
 * Returns the memory block to the offload memory pool. The pointer must
 * have been obtained from offload malloc.
 *
 * @param ptr   [in] Address returned by offload malloc.
 * @param flags [in] optional flags
 */
void offload_free(uint64_t ptr, uint64_t flags);

/**
 * @brief Batch copy sparse data from host to device or from device to host.
 *
 * Submits a batch of h2d or d2h copy requests. Each request copies
 * data from srcPtrs[i] to dstPtrs[i] with length lenPtrs[i]. The copy is
 * executed asynchronously on the device stream.
 *
 * @param srcPtrs   [in] Array of source addresses.
 * @param dstPtrs   [in] Array of destination addresses.
 * @param lenPtrs   [in] Array of byte counts to copy for each pair.
 * @param sizePtr   [in] Pointer to the number of entries in the arrays above.
 * @param deviceId  [in] Device ID to perform the copy on.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_sparse_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t sizePtr, uint16_t deviceId);

/**
 * @brief Run the LRU resident compact kernel on device.
 *
 * Mirrors the CPU lru_resident_compact in vllm-ascend sfa_kv_offload. All
 * pointers must be device (NPU HBM) addresses. The per-core workspace buffers
 * (token_mark_workspace, token_pos_workspace, epochs) must be allocated with
 * first dim = LRU_COMPACT_BLOCK_DIM (=8), i.e. [8, max_token] and [8].
 *
 * @param req_ids                [in] device ptr, int64[num_reqs]
 * @param last_req_ids           [in/out] device ptr, int64[num_reqs], persistent state
 * @param topk_indices           [in] device ptr, int32[num_reqs * topk]
 * @param stable_prefix_lens      [in] device ptr, int32[num_reqs]; token >= stable_prefix_lens[row] is a speculative
 *                                  suffix that may have been overwritten in the CPU KV pool and is invalidated
 * @param slot_to_token          [in/out] device ptr, int32[num_reqs * capacity], persistent state
 * @param lru_slots             [in/out] device ptr, int32[num_reqs * capacity], persistent state
 * @param current_slots          [out] device ptr, int32[num_reqs * topk]
 * @param miss_count             [out] device ptr, int32[num_reqs]
 * @param miss_tokens            [out] device ptr, int32[num_reqs * topk]
 * @param miss_slots             [out] device ptr, int32[num_reqs * topk]
 * @param token_mark_workspace   [in/out] device ptr, int32[8 * max_token], per-core workspace
 * @param token_pos_workspace    [in/out] device ptr, int32[8 * max_token], per-core workspace
 * @param epochs                 [in/out] device ptr, int32[8], per-core workspace
 * @param num_reqs / topk / capacity / max_token  problem dimensions
 * @param deviceId               [in] NPU device id
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_lru_resident_compact(uint64_t req_ids, uint64_t last_req_ids, uint64_t topk_indices,
                                     uint64_t stable_prefix_lens, uint64_t slot_to_token, uint64_t lru_slots,
                                     uint64_t current_slots, uint64_t miss_count, uint64_t miss_tokens,
                                     uint64_t miss_slots, uint64_t token_mark_workspace, uint64_t token_pos_workspace,
                                     uint64_t epochs, int64_t num_reqs, int64_t topk, int64_t capacity,
                                     int64_t max_token, uint16_t deviceId);

/**
 * @brief Run the compute_lru_resident_addrs kernel on device.
 *
 * Mirrors the CPU compute_lru_resident_addrs in vllm-ascend sfa_kv_offload.
 * Produces gvas_buffer / addr_buffer / size_buffer / num_tokens_buffer with the
 * exact layout consumed by offload_sparse_copy (first half = K, second half = V).
 *
 * @param miss_count / miss_tokens / miss_slots  outputs of offload_lru_resident_compact
 * @param block_table            [in] device ptr, int32[num_reqs * max_num_blocks]
 * @param gvas_buffer            [out] device ptr, int64[num_reqs * topk * 2]
 * @param addr_buffer            [out] device ptr, int64[num_reqs * topk * 2]
 * @param size_buffer            [out] device ptr, int32[num_reqs * topk * 2]
 * @param num_tokens_buffer      [out] device ptr, int32[1]
 * @param block_size / token_size_bytes_k / token_size_bytes_v   token geometry (bytes)
 * @param gvas_k_base / gvas_v_base  source (CPU offload pool) base addresses
 * @param addr_k_base / addr_v_base  destination (NPU resident buffer) base addresses
 * @param resident_capacity      resident buffer capacity (per req)
 * @param num_reqs / topk / max_num_blocks  problem dimensions
 * @param deviceId               [in] NPU device id
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_compute_lru_resident_addrs(uint64_t miss_count, uint64_t miss_tokens, uint64_t miss_slots,
                                           uint64_t block_table, uint64_t gvas_buffer, uint64_t addr_buffer,
                                           uint64_t size_buffer, uint64_t num_tokens_buffer, int32_t block_size,
                                           int32_t token_size_bytes_k, int32_t token_size_bytes_v,
                                           int64_t gvas_k_base, int64_t gvas_v_base, int64_t addr_k_base,
                                           int64_t addr_v_base, int32_t resident_capacity, int64_t num_reqs,
                                           int64_t topk, int64_t max_num_blocks, uint16_t deviceId);

#ifdef __cplusplus
}
#endif

#endif //__MEMFABRIC_ACC_OFFLOAD_H__