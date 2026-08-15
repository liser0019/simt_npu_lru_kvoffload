#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import ctypes

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(current_dir, 'lib')
for lib in ["libmf_hybm_core.so", "libmf_smem.so", "libmf_acc_offload.so"]:
    ctypes.CDLL(os.path.join(libs_path, lib))


def get_include_path():
    package_dir = os.path.dirname(__file__)
    include_path = os.path.join(package_dir, 'include')
    return os.path.abspath(include_path)


def get_lib_path():
    package_dir = os.path.dirname(__file__)
    lib_path = os.path.join(package_dir, 'lib')
    return os.path.abspath(lib_path)


from _pymf_transfer import (
    TransferEngine,
    TransferOpcode,
    create_config_store
)
from _pymf_hybrid import (
    bm,
    shm,
    initialize,
    uninitialize,
    set_log_level,
    set_extern_logger,
    get_last_err_msg,
    set_conf_store_tls,
    set_conf_store_tls_key,
    get_and_clear_last_err_msg
)
from _pymf_acc_offload import offload
from mf_acc_offload import (empty, get_device_address, sparse_copy, lru_resident_compact,
                            compute_lru_resident_addrs,
                            get_sparse_kv_plan_workspace_size,
                            sparse_kv_load_runtime,
                            get_sparse_kv_fsa_row_map_workspace_size,
                            get_sparse_kv_fsa_plan_row_stride,
                            sparse_kv_plan_fsa_runtime)

offload.empty = empty
offload.get_device_address = get_device_address
offload.sparse_copy = sparse_copy
offload.lru_resident_compact = lru_resident_compact
offload.compute_lru_resident_addrs = compute_lru_resident_addrs
offload.get_sparse_kv_plan_workspace_size = get_sparse_kv_plan_workspace_size
offload.sparse_kv_load_runtime = sparse_kv_load_runtime
offload.get_sparse_kv_fsa_row_map_workspace_size = \
    get_sparse_kv_fsa_row_map_workspace_size
offload.get_sparse_kv_fsa_plan_row_stride = get_sparse_kv_fsa_plan_row_stride
offload.sparse_kv_plan_fsa_runtime = sparse_kv_plan_fsa_runtime


__all__ = [
    'TransferEngine',
    'TransferOpcode',
    'create_config_store',
    'bm',
    'shm',
    'offload',
    'initialize',
    'uninitialize',
    'set_log_level',
    'set_extern_logger',
    'get_last_err_msg',
    'set_conf_store_tls',
    'set_conf_store_tls_key',
    'get_and_clear_last_err_msg'
]
__all__ += ['get_include_path', 'get_lib_path']
