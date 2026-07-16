/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ACC_LINKS_ACC_TCP_PORT_UTIL_H
#define ACC_LINKS_ACC_TCP_PORT_UTIL_H

#include <cstdint>

namespace ock {
namespace acc {

/**
 * @brief Probe a free TCP port in [minPort, maxPort] by binding a socket.
 *
 * The probe socket is kept bound (returned via @p sockfd) so the caller can
 * reserve the port (e.g. a client that does not start a real listener) or
 * close it immediately before starting a real listener on the same port.
 *
 * Any previously held fd in @p sockfd is closed first to avoid fd leakage on
 * repeated calls.
 *
 * @param sockfd   [in,out] holds an already-bound fd on success (>=0), -1 on
 *                 failure; an incoming valid fd is closed first.
 * @param minPort  [in] lower bound of the port range (inclusive), must be > 0.
 * @param maxPort  [in] upper bound of the port range (inclusive), >= minPort.
 * @return the bound port number on success, or 0 on failure.
 */
uint16_t AccFindAvailableTcpPort(int &sockfd, uint16_t minPort, uint16_t maxPort);

} // namespace acc
} // namespace ock

#endif // ACC_LINKS_ACC_TCP_PORT_UTIL_H
