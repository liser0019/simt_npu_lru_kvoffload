# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

include_directories(
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/include/host
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/common
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/allocator
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/launch
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/operators
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem_bm
        ${PROJECT_SMEM_SRC_BASE}/csrc/common
        ${PROJECT_SMEM_SRC_BASE}/csrc/net
        ${PROJECT_HYBM_SRC_BASE}/include
        ${PROJECT_UTIL_SRC_BASE}/ptracer/include
        ${PROJECT_ACCLINKS_SRC_BASE}/include
)

# acc_offload: glob all, then EXCLUDE operators/ and launch/ (AscendC kernel + torch_npu adapter,
# compiled at install time via try_install_extend, see script/run_pkg_maker/install.sh)
file(GLOB_RECURSE ACC_OFFLOAD_SRC
        ${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/*.cpp
)
list(FILTER ACC_OFFLOAD_SRC EXCLUDE REGEX "${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/operators/.*")
list(FILTER ACC_OFFLOAD_SRC EXCLUDE REGEX "${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/launch/acc_offload_operators_launch.cpp")
list(FILTER ACC_OFFLOAD_SRC EXCLUDE REGEX "${PROJECT_ACC_OFFLOAD_SRC_BASE}/csrc/python_wrapper/.*")

add_library(acc_offload_objects OBJECT ${ACC_OFFLOAD_SRC})
target_link_libraries(acc_offload_objects PUBLIC config_store_object)

target_compile_options(acc_offload_objects PRIVATE -fPIC)

add_library(acc_offload_static STATIC $<TARGET_OBJECTS:acc_offload_objects> $<TARGET_OBJECTS:config_store_object>)
set_target_properties(acc_offload_static PROPERTIES OUTPUT_NAME "mf_acc_offload")
target_link_libraries(acc_offload_static PUBLIC pthread acc_tcp_net_static hybmm_static smem_static)

add_library(acc_offload_shared SHARED $<TARGET_OBJECTS:acc_offload_objects> $<TARGET_OBJECTS:config_store_object>)
set_target_properties(acc_offload_shared PROPERTIES
    OUTPUT_NAME "mf_acc_offload"
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_link_libraries(acc_offload_shared PUBLIC pthread acc_tcp_net_static hybmm_shared smem_shared)

if ((BUILD_PYTHON STREQUAL "ON") AND (NOT BUILD_UT STREQUAL "ON"))
    add_subdirectory(python_wrapper)
endif ()

#install
file(GLOB HEADER_FILES ../include/host/*.h)

install(FILES ${HEADER_FILES} DESTINATION ${TARGET_INSTALL_DIR}/acc_offload/include/host)

install(TARGETS acc_offload_shared
        LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/acc_offload/lib64
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)