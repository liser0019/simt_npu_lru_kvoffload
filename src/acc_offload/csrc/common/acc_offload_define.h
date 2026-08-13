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

#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_DEFINE_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_DEFINE_H

#include <cstdint>
#include "acc_offload_logger.h"

namespace ock {
namespace offload {

#define OFFLOAD_OK                               (0)
#define OFFLOAD_ERROR                            (-1)
#define OFFLOAD_UNLOAD                           (-2)
#define OFFLOAD_FUNCTION_FAILED                  (-3)

#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1) != 0)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0) != 0)
#endif

#define OFFLOAD_API __attribute__((visibility("default")))

#define DL_LOAD_SYM_OPTIONAL(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)              \
    do {                                                                                               \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                           \
        if ((TARGET_FUNC_VAR) == nullptr) {                                                            \
            OFFLOAD_LOG_WARN("Failed to call dlsym to load " << (SYMBOL_NAME) << ", error: " << dlerror()); \
        }                                                                                              \
    } while (0)

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_DEFINE_H