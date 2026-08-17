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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_STORE_PORT_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_STORE_PORT_H

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ock {
namespace offload {
namespace internal {

constexpr const char *ACC_OFFLOAD_PORT_BASE_ENV = "MF_ACC_OFFLOAD_PORT_BASE";
constexpr uint32_t ACC_OFFLOAD_DEFAULT_PORT_BASE = 8500U;
constexpr uint32_t ACC_OFFLOAD_MAX_TCP_PORT = 65535U;

inline bool ResolveAccOffloadStorePort(uint32_t deviceId, uint32_t worldSize, uint16_t &port,
                                       std::string &error)
{
    if (worldSize == 0U) {
        error = "worldSize must be greater than zero";
        return false;
    }

    uint32_t portBase = ACC_OFFLOAD_DEFAULT_PORT_BASE;
    const char *value = std::getenv(ACC_OFFLOAD_PORT_BASE_ENV);
    if (value != nullptr) {
        if (*value == '\0') {
            error = "environment value is empty";
            return false;
        }
        for (const char *cursor = value; *cursor != '\0'; ++cursor) {
            if (*cursor < '0' || *cursor > '9') {
                error = "environment value must contain decimal digits only: '" + std::string(value) + "'";
                return false;
            }
        }

        char *end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed == 0UL || parsed > ACC_OFFLOAD_MAX_TCP_PORT) {
            error = "environment value is outside the valid TCP port range: '" + std::string(value) + "'";
            return false;
        }
        portBase = static_cast<uint32_t>(parsed);
    }

    const uint32_t portOffset = deviceId / worldSize;
    if (portOffset > ACC_OFFLOAD_MAX_TCP_PORT - portBase) {
        error = "port base " + std::to_string(portBase) + " plus device/world offset " +
                std::to_string(portOffset) + " exceeds 65535";
        return false;
    }

    port = static_cast<uint16_t>(portBase + portOffset);
    error.clear();
    return true;
}

} // namespace internal
} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_STORE_PORT_H
