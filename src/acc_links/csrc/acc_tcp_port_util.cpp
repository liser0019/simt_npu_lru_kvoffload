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
#include "acc_tcp_port_util.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <random>
#include <sys/socket.h>
#include <unistd.h>

#include "acc_includes.h"
#include "acc_tcp_common.h"

namespace ock {
namespace acc {
namespace {

constexpr int MAX_ATTEMPTS = 1000;

int BindTcpPortV4(int &sockfd, uint16_t port)
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (::bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        int err = errno;
        LOG_ERROR("bind ipv4 tcp port failed, port=" << port << ", errno=" << err << ", errstr=" << std::strerror(err));
        SafeCloseFd(sockfd);
        return -1;
    }
    return 0;
}

int BindTcpPortV6(int &sockfd, uint16_t port)
{
    sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd < 0) {
        return -1;
    }
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htons(port);
    addr.sin6_addr = in6addr_any;
    if (::bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        int err = errno;
        LOG_ERROR("bind ipv6 tcp port failed, port=" << port << ", errno=" << err << ", errstr=" << std::strerror(err));
        SafeCloseFd(sockfd);
        return -1;
    }
    return 0;
}

bool SupportsIpv6() noexcept
{
    int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    SafeCloseFd(fd);
    return true;
}

uint64_t BuildProbeSeed() noexcept
{
    constexpr int offsetBit = 32;
    uint64_t seed = 1;
    seed |= static_cast<uint64_t>(getpid()) << offsetBit;
    seed |= static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) & 0xFFFFFFFFULL;
    return seed;
}

std::mt19937_64 &RandomEngine()
{
    static std::mt19937_64 gen(BuildProbeSeed());
    return gen;
}
} // namespace

uint16_t AccFindAvailableTcpPort(int &sockfd, uint16_t minPort, uint16_t maxPort)
{
    if (sockfd >= 0) {
        SafeCloseFd(sockfd);
    }
    if (minPort == 0 || maxPort < minPort) {
        LOG_ERROR("invalid port range, minPort=" << minPort << ", maxPort=" << maxPort);
        return 0;
    }

    std::uniform_int_distribution<uint32_t> dis(minPort, maxPort);
    const bool supportsIpv6 = SupportsIpv6();
    auto &gen = RandomEngine();

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        uint16_t port = static_cast<uint16_t>(dis(gen));
        if (BindTcpPortV4(sockfd, port) == 0) {
            return port;
        }
        if (supportsIpv6 && BindTcpPortV6(sockfd, port) == 0) {
            return port;
        }
    }
    LOG_ERROR("no available tcp port found in range [" << minPort << "," << maxPort << "]");
    return 0;
}
} // namespace acc
} // namespace ock
