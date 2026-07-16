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
#include <cctype>
#include <cstdlib>
#include <string>

#include "acc_tcp_port_util.h"
#include "adapter_logger.h"
#include "mf_env_define.h"
#include "mf_env_util.h"
#include "smem.h"
#include "transfer_util.h"

namespace ock {
namespace adapter {

uint16_t AccFindAvailableTcpPortAdapter(int &sockfd)
{
    uint16_t minPort = DEFAULT_PORT_START;
    uint16_t maxPort = DEFAULT_PORT_MAX;
    GetConfigStorePortRange(minPort, maxPort);
    return ock::acc::AccFindAvailableTcpPort(sockfd, minPort, maxPort);
}

void ParseIpPortFromUniqueId(const std::string &uniqueId, std::string &ip, uint16_t &port)
{
    ip.clear();
    port = 0;
    auto pos = uniqueId.rfind(':');
    if (pos == std::string::npos) {
        ip = uniqueId;
        ADAPTER_LOG_INFO("uniqueId='" << uniqueId << "' has no port suffix, auto-select port");
        return;
    }
    if (pos + 1 >= uniqueId.size()) {
        ip = uniqueId.substr(0, pos);
        ADAPTER_LOG_INFO("uniqueId='" << uniqueId << "' has empty port suffix, auto-select port");
        return;
    }
    const std::string portStr = uniqueId.substr(pos + 1);
    for (char c : portStr) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            ip = uniqueId;
            ADAPTER_LOG_WARN("uniqueId='" << uniqueId << "' port suffix '" << portStr
                                          << "' is non-numeric, treat whole as ip and auto-select port");
            return;
        }
    }
    unsigned long parsed = 0;
    try {
        parsed = std::stoul(portStr);
    } catch (...) {
        ip = uniqueId;
        ADAPTER_LOG_WARN("uniqueId='" << uniqueId << "' port suffix '" << portStr
                                      << "' is out of range, auto-select port");
        return;
    }
    if (parsed > 65535UL) {
        ip = uniqueId;
        ADAPTER_LOG_WARN("uniqueId='" << uniqueId << "' port " << parsed << " > 65535, auto-select port");
        return;
    }
    ip = uniqueId.substr(0, pos);
    port = static_cast<uint16_t>(parsed);
}

std::string BuildSessionId(const std::string &ip, uint16_t port)
{
    return ip + ":" + std::to_string(port);
}

void GetConfigStorePortRange(uint16_t &minPort, uint16_t &maxPort)
{
    minPort = DEFAULT_PORT_START;
    maxPort = DEFAULT_PORT_MAX;

    // Read the env var live (each call) so the range is adjustable at runtime
    // and unit-testable via setenv (the static-initialized ock::mf::env::* consts
    // cache the value once at program start and cannot be changed later).
    const auto parsePort = [](const char *envName, uint16_t def, uint16_t &out) {
        out = def;
        const char *val = std::getenv(envName);
        if (val == nullptr || val[0] == '\0') {
            return;
        }
        uint32_t parsed = 0;
        if (!ock::mf::MfEnvUtil::GetUint(std::string(val), parsed) || parsed > 65535UL) {
            ADAPTER_LOG_WARN("invalid port env " << envName << "='" << val << "', fallback to " << def);
            return;
        }
        out = static_cast<uint16_t>(parsed);
    };

    uint16_t startP = DEFAULT_PORT_START;
    uint16_t endP = DEFAULT_PORT_MAX;
    parsePort("MF_CONFIG_STORE_PORT_START", DEFAULT_PORT_START, startP);
    parsePort("MF_CONFIG_STORE_PORT_END", DEFAULT_PORT_MAX, endP);
    if (startP > endP) {
        ADAPTER_LOG_WARN("invalid port range start=" << startP << " > end=" << endP << ", fallback to default ["
                                                     << DEFAULT_PORT_START << "," << DEFAULT_PORT_MAX << "]");
        return;
    }
    minPort = startP;
    maxPort = endP;
}

int32_t pytransfer_create_config_store(const char *storeUrl)
{
    const char *shmem_level = std::getenv("SHMEM_LOG_LEVEL");
    const char *mf_level = std::getenv("ASCEND_MF_LOG_LEVEL");
    if (shmem_level == nullptr && mf_level != nullptr && strlen(mf_level) == 1) {
        unsigned char c = static_cast<unsigned char>(mf_level[0]);
        if (std::isdigit(c)) {
            int level = c - '0';
            smem_set_log_level(level);
        }
    }
    // default: disable tls
    smem_set_conf_store_tls(false, nullptr, 0);
    int ret = smem_create_config_store(storeUrl, SMEM_STORE_SKIP_RECOVER);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_create_config_store happen error, ret=" << ret);
    }
    return ret;
}

int32_t pytransfer_set_log_level(int level)
{
    int ret = smem_set_log_level(level);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_set_log_level happen error, ret=" << ret);
        return ret;
    }
    return 0;
}

int32_t pytransfer_set_conf_store_tls(bool enable, std::string &tls_info)
{
    return smem_set_conf_store_tls(enable, tls_info.c_str(), tls_info.size());
}

} // namespace adapter
} // namespace ock