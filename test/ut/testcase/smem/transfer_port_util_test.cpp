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

#include <unistd.h>

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "transfer_util.h"

using namespace ock::adapter;

namespace {
constexpr uint16_t K_PORT_ZERO = 0;
constexpr uint16_t K_DEF_MIN = DEFAULT_PORT_START; // 9000
constexpr uint16_t K_DEF_MAX = DEFAULT_PORT_MAX;   // 65535
constexpr uint16_t K_ENV_MIN = 49300;
constexpr uint16_t K_ENV_MAX = 49350;
constexpr int K_INVALID_FD = -1;
} // namespace

class ParseIpPortFromUniqueIdTest : public testing::Test {};

TEST_F(ParseIpPortFromUniqueIdTest, Ipv4WithPort)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("127.0.0.1:10000", ip, port);
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, 10000);
}

TEST_F(ParseIpPortFromUniqueIdTest, Ipv4OnlyAutoSelect)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("127.0.0.1", ip, port);
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, PortZeroMeansAutoSelect)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("127.0.0.1:0", ip, port);
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, TrailingColonStripped)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("127.0.0.1:", ip, port);
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, PortMaxBoundary)
{
    std::string ip;
    uint16_t port = 0;
    ParseIpPortFromUniqueId("10.0.0.1:65535", ip, port);
    EXPECT_EQ(ip, "10.0.0.1");
    EXPECT_EQ(port, 65535);
}

TEST_F(ParseIpPortFromUniqueIdTest, PortOverflowTreatedAsIp)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("10.0.0.1:65536", ip, port);
    EXPECT_EQ(ip, "10.0.0.1:65536");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, NonNumericSuffixTreatedAsIp)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("host:abc", ip, port);
    EXPECT_EQ(ip, "host:abc");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, HugePortValueThrowsHandled)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("10.0.0.1:99999999999999999999", ip, port);
    EXPECT_EQ(ip, "10.0.0.1:99999999999999999999");
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, EmptyInput)
{
    std::string ip = "x";
    uint16_t port = 1;
    ParseIpPortFromUniqueId("", ip, port);
    EXPECT_TRUE(ip.empty());
    EXPECT_EQ(port, K_PORT_ZERO);
}

TEST_F(ParseIpPortFromUniqueIdTest, Ipv6BracketedWithPort)
{
    std::string ip;
    uint16_t port = 0;
    ParseIpPortFromUniqueId("[::1]:8080", ip, port);
    EXPECT_EQ(ip, "[::1]");
    EXPECT_EQ(port, 8080);
}

TEST_F(ParseIpPortFromUniqueIdTest, Ipv6BracketedAutoSelect)
{
    std::string ip;
    uint16_t port = 1;
    ParseIpPortFromUniqueId("[::1]", ip, port);
    EXPECT_EQ(ip, "[::1]");
    EXPECT_EQ(port, K_PORT_ZERO);
}

class GetConfigStorePortRangeTest : public testing::Test {
public:
    void SetUp() override
    {
        (void)unsetenv("MF_CONFIG_STORE_PORT_START");
        (void)unsetenv("MF_CONFIG_STORE_PORT_END");
    }

    void TearDown() override
    {
        (void)unsetenv("MF_CONFIG_STORE_PORT_START");
        (void)unsetenv("MF_CONFIG_STORE_PORT_END");
    }
};

TEST_F(GetConfigStorePortRangeTest, DefaultsWhenEnvUnset)
{
    uint16_t minP = 1;
    uint16_t maxP = 1;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(minP, K_DEF_MIN);
    EXPECT_EQ(maxP, K_DEF_MAX);
}

TEST_F(GetConfigStorePortRangeTest, RespectsEnvRange)
{
    (void)setenv("MF_CONFIG_STORE_PORT_START", "49300", 1);
    (void)setenv("MF_CONFIG_STORE_PORT_END", "49350", 1);

    uint16_t minP = 0;
    uint16_t maxP = 0;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(minP, K_ENV_MIN);
    EXPECT_EQ(maxP, K_ENV_MAX);
}

TEST_F(GetConfigStorePortRangeTest, InvalidEnvValueFallsBack)
{
    (void)setenv("MF_CONFIG_STORE_PORT_START", "notanumber", 1);

    uint16_t minP = 0;
    uint16_t maxP = 0;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(minP, K_DEF_MIN);
    EXPECT_EQ(maxP, K_DEF_MAX);
}

TEST_F(GetConfigStorePortRangeTest, EnvZeroFallsBack)
{
    (void)setenv("MF_CONFIG_STORE_PORT_START", "0", 1);

    uint16_t minP = 0;
    uint16_t maxP = 0;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(minP, K_DEF_MIN);
}

TEST_F(GetConfigStorePortRangeTest, EnvOverflowFallsBack)
{
    (void)setenv("MF_CONFIG_STORE_PORT_END", "99999", 1);

    uint16_t minP = 0;
    uint16_t maxP = 0;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(maxP, K_DEF_MAX);
}

TEST_F(GetConfigStorePortRangeTest, ReversedRangeFallsBack)
{
    (void)setenv("MF_CONFIG_STORE_PORT_START", "20000", 1);
    (void)setenv("MF_CONFIG_STORE_PORT_END", "10000", 1);

    uint16_t minP = 0;
    uint16_t maxP = 0;
    GetConfigStorePortRange(minP, maxP);
    EXPECT_EQ(minP, K_DEF_MIN);
    EXPECT_EQ(maxP, K_DEF_MAX);
}

class AccFindAvailableTcpPortAdapterTest : public testing::Test {
public:
    void SetUp() override
    {
        (void)unsetenv("MF_CONFIG_STORE_PORT_START");
        (void)unsetenv("MF_CONFIG_STORE_PORT_END");
        sockfd_ = K_INVALID_FD;
    }

    void TearDown() override
    {
        if (sockfd_ >= 0) {
            close(sockfd_);
            sockfd_ = K_INVALID_FD;
        }
        (void)unsetenv("MF_CONFIG_STORE_PORT_START");
        (void)unsetenv("MF_CONFIG_STORE_PORT_END");
    }

    int sockfd_ = K_INVALID_FD;
};

TEST_F(AccFindAvailableTcpPortAdapterTest, FindsPortInDefaultRange)
{
    uint16_t port = AccFindAvailableTcpPortAdapter(sockfd_);
    EXPECT_GE(port, K_DEF_MIN);
    EXPECT_LE(port, K_DEF_MAX);
    EXPECT_GE(sockfd_, 0);
}

TEST_F(AccFindAvailableTcpPortAdapterTest, RespectsEnvRange)
{
    (void)setenv("MF_CONFIG_STORE_PORT_START", "49300", 1);
    (void)setenv("MF_CONFIG_STORE_PORT_END", "49350", 1);

    uint16_t port = AccFindAvailableTcpPortAdapter(sockfd_);
    EXPECT_GE(port, K_ENV_MIN);
    EXPECT_LE(port, K_ENV_MAX);
}
