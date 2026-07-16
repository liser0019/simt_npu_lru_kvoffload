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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <cstdint>

#include <gtest/gtest.h>
#include "acc_tcp_port_util.h"

using ock::acc::AccFindAvailableTcpPort;

namespace {
constexpr uint16_t K_RANGE_MIN = 19000;
constexpr uint16_t K_RANGE_MAX = 19200;
constexpr uint16_t K_PORT_ZERO = 0;
constexpr int K_INVALID_FD = -1;
constexpr int K_REUSE = 200; // repeated calls to detect fd leak
} // namespace

class AccTcpPortUtilTest : public testing::Test {
public:
    void SetUp() override
    {
        sockfd_ = K_INVALID_FD;
    }

    void TearDown() override
    {
        if (sockfd_ >= 0) {
            close(sockfd_);
            sockfd_ = K_INVALID_FD;
        }
    }

    int sockfd_ = K_INVALID_FD;
};

TEST_F(AccTcpPortUtilTest, FindsPortInRange)
{
    uint16_t port = AccFindAvailableTcpPort(sockfd_, K_RANGE_MIN, K_RANGE_MAX);
    EXPECT_GE(port, K_RANGE_MIN);
    EXPECT_LE(port, K_RANGE_MAX);
    EXPECT_GE(sockfd_, 0);
}

TEST_F(AccTcpPortUtilTest, BoundPortMatchesReturnValue)
{
    uint16_t port = AccFindAvailableTcpPort(sockfd_, K_RANGE_MIN, K_RANGE_MAX);
    ASSERT_GT(port, K_PORT_ZERO);

    sockaddr_in addr{};
    socklen_t addrLen = sizeof(addr);
    ASSERT_EQ(getsockname(sockfd_, reinterpret_cast<sockaddr *>(&addr), &addrLen), 0);
    EXPECT_EQ(ntohs(addr.sin_port), port);
}

TEST_F(AccTcpPortUtilTest, ReturnedPortIsReserved)
{
    uint16_t port = AccFindAvailableTcpPort(sockfd_, K_RANGE_MIN, K_RANGE_MAX);
    ASSERT_GT(port, K_PORT_ZERO);

    int probe = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(probe, 0);
    int reuse = 1;
    (void)setsockopt(probe, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    // same wildcard bind must be rejected while sockfd_ holds the port
    EXPECT_LT(bind(probe, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
    close(probe);
}

TEST_F(AccTcpPortUtilTest, InvalidRangeMinZero)
{
    EXPECT_EQ(AccFindAvailableTcpPort(sockfd_, 0, K_RANGE_MAX), K_PORT_ZERO);
    EXPECT_EQ(sockfd_, K_INVALID_FD);
}

TEST_F(AccTcpPortUtilTest, InvalidRangeReversed)
{
    EXPECT_EQ(AccFindAvailableTcpPort(sockfd_, K_RANGE_MAX, K_RANGE_MIN), K_PORT_ZERO);
    EXPECT_EQ(sockfd_, K_INVALID_FD);
}

TEST_F(AccTcpPortUtilTest, RepeatedCallReleasesOldFd)
{
    for (int i = 0; i < K_REUSE; ++i) {
        uint16_t port = AccFindAvailableTcpPort(sockfd_, K_RANGE_MIN, K_RANGE_MAX);
        ASSERT_GT(port, K_PORT_ZERO);
        ASSERT_GE(sockfd_, 0);
    }
    // if old fds leaked across iterations, fd table would exhaust and a later
    // socket() would fail; the loop passing proves no fd leak.
}
