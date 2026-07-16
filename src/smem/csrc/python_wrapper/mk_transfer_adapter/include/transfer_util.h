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
#ifndef PYTRANSFER_UITL_H
#define PYTRANSFER_UITL_H

#include <cstdint>
#include <string>

namespace ock {
namespace adapter {

#define STR(x)  #x
#define STR2(x) STR(x)

// Default config-store port range, consistent with smem HA layer
// (network_endpoint_util.h kDefaultStartPort/kDefaultMaxPort).
constexpr uint16_t DEFAULT_PORT_START = 9000;
constexpr uint16_t DEFAULT_PORT_MAX = 65535;
constexpr uint32_t PORT_SELECT_MAX_RETRY = 64;

int32_t pytransfer_create_config_store(const char *storeUrl);

/**
 * @brief Resolve the config-store port range from env
 *        MF_CONFIG_STORE_PORT_START / MF_CONFIG_STORE_PORT_END.
 *
 * @param minPort [out] lower bound (inclusive), DEFAULT_PORT_START if unset.
 * @param maxPort [out] upper bound (inclusive), DEFAULT_PORT_MAX if unset.
 */
void GetConfigStorePortRange(uint16_t &minPort, uint16_t &maxPort);

/**
 * @brief Probe a free TCP port using the acc_links utility, with the port range
 *        resolved from env (see GetConfigStorePortRange).
 *
 * The probe socket is kept bound in @p sockfd on success (caller decides
 * whether to keep or close it). Returns 0 on failure.
 */
uint16_t AccFindAvailableTcpPortAdapter(int &sockfd);

/**
 * @brief Parse "IP:PORT" / "IP" / "IP:0" into ip and port.
 *
 * port==0 means "auto-select" (no explicit port, or "0", or non-numeric /
 * out-of-range suffix). On such cases the whole @p uniqueId is returned as @p ip.
 */
void ParseIpPortFromUniqueId(const std::string &uniqueId, std::string &ip, uint16_t &port);

/** @brief Build the session id "ip:port" (IPv6 keeps its brackets). */
std::string BuildSessionId(const std::string &ip, uint16_t port);

int32_t pytransfer_set_log_level(int level);

int32_t pytransfer_set_conf_store_tls(bool enable, std::string &tls_info);

} // namespace adapter
} // namespace ock
#endif // PYTRANSFER_UITL_H
