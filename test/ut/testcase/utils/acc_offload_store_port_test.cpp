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

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>
#include "acc_offload_store_port.h"

using ock::offload::internal::ACC_OFFLOAD_PORT_BASE_ENV;
using ock::offload::internal::ResolveAccOffloadStorePort;

class AccOffloadStorePortTest : public testing::Test {
public:
    void SetUp() override
    {
        (void)unsetenv(ACC_OFFLOAD_PORT_BASE_ENV);
    }

    void TearDown() override
    {
        (void)unsetenv(ACC_OFFLOAD_PORT_BASE_ENV);
    }
};

TEST_F(AccOffloadStorePortTest, UsesBackwardCompatibleDefault)
{
    uint16_t port = 0U;
    std::string error;
    ASSERT_TRUE(ResolveAccOffloadStorePort(0U, 1U, port, error));
    EXPECT_EQ(port, 8500U);
    EXPECT_TRUE(error.empty());
}

TEST_F(AccOffloadStorePortTest, UsesConfiguredBaseAndPreservesOffset)
{
    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "18500", 1), 0);
    uint16_t port = 0U;
    std::string error;
    ASSERT_TRUE(ResolveAccOffloadStorePort(4U, 2U, port, error));
    EXPECT_EQ(port, 18502U);
    EXPECT_TRUE(error.empty());
}

TEST_F(AccOffloadStorePortTest, RejectsNonDecimalValue)
{
    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "abc", 1), 0);
    uint16_t port = 0U;
    std::string error;
    EXPECT_FALSE(ResolveAccOffloadStorePort(0U, 1U, port, error));
    EXPECT_FALSE(error.empty());
}

TEST_F(AccOffloadStorePortTest, RejectsOutOfRangeBase)
{
    uint16_t port = 0U;
    std::string error;

    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "0", 1), 0);
    EXPECT_FALSE(ResolveAccOffloadStorePort(0U, 1U, port, error));

    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "70000", 1), 0);
    EXPECT_FALSE(ResolveAccOffloadStorePort(0U, 1U, port, error));
}

TEST_F(AccOffloadStorePortTest, RejectsFinalPortOverflow)
{
    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "65535", 1), 0);
    uint16_t port = 0U;
    std::string error;
    EXPECT_FALSE(ResolveAccOffloadStorePort(1U, 1U, port, error));
    EXPECT_NE(error.find("exceeds 65535"), std::string::npos);
}

TEST_F(AccOffloadStorePortTest, AcceptsMaximumFinalPort)
{
    ASSERT_EQ(setenv(ACC_OFFLOAD_PORT_BASE_ENV, "65534", 1), 0);
    uint16_t port = 0U;
    std::string error;
    ASSERT_TRUE(ResolveAccOffloadStorePort(1U, 1U, port, error));
    EXPECT_EQ(port, 65535U);
}
