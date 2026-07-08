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

#ifdef __cplusplus
}
#endif

#endif //__MEMFABRIC_ACC_OFFLOAD_H__