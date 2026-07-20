#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ctypes
import torch
from _pymf_acc_offload import offload

sparse_copy_impl = offload.sparse_copy
lru_resident_compact_impl = offload.lru_resident_compact
compute_lru_resident_addrs_impl = offload.compute_lru_resident_addrs


def empty(sizes, dtype=None, pin_memory=False):
    if dtype is None:
        dtype = torch.bfloat16

    numel = 1
    for size in sizes:
        numel *= size
    element_size = numel * torch.tensor([], dtype=dtype).element_size()
    ptr = offload.malloc(element_size)
    if ptr == 0:
        raise Exception("malloc failed")
    buf = (ctypes.c_int8 * element_size).from_address(ptr)
    return torch.frombuffer(buf, dtype=dtype).reshape(sizes)


def sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId):
    return sparse_copy_impl(srcPtrs.data_ptr(), dstPtrs.data_ptr(), lenPtrs.data_ptr(), sizePtr.data_ptr(),
                            deviceId.index)


def lru_resident_compact(req_ids, last_req_ids, topk_indices, stable_prefix_lens, slot_to_token, lru_slots,
                         current_slots, miss_count, miss_tokens, miss_slots, token_mark_workspace,
                         token_pos_workspace, epochs, num_reqs, topk, capacity, max_token, deviceId):
    return lru_resident_compact_impl(
        req_ids.data_ptr(), last_req_ids.data_ptr(), topk_indices.data_ptr(),
        stable_prefix_lens.data_ptr(), slot_to_token.data_ptr(), lru_slots.data_ptr(),
        current_slots.data_ptr(), miss_count.data_ptr(), miss_tokens.data_ptr(), miss_slots.data_ptr(),
        token_mark_workspace.data_ptr(), token_pos_workspace.data_ptr(), epochs.data_ptr(), num_reqs, topk,
        capacity, max_token, deviceId.index)


def compute_lru_resident_addrs(miss_count, miss_tokens, miss_slots, block_table, gvas_buffer, addr_buffer,
                               size_buffer, num_tokens_buffer, block_size, token_size_bytes_k,
                               token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
                               resident_capacity, num_reqs, topk, max_num_blocks, deviceId):
    return compute_lru_resident_addrs_impl(
        miss_count.data_ptr(), miss_tokens.data_ptr(), miss_slots.data_ptr(), block_table.data_ptr(),
        gvas_buffer.data_ptr(), addr_buffer.data_ptr(), size_buffer.data_ptr(), num_tokens_buffer.data_ptr(),
        block_size, token_size_bytes_k, token_size_bytes_v, gvas_k_base, gvas_v_base, addr_k_base, addr_v_base,
        resident_capacity, num_reqs, topk, max_num_blocks, deviceId.index)