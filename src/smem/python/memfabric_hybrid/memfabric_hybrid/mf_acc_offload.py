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

get_device_address_impl = offload.get_device_address
sparse_copy_impl = offload.sparse_copy
lru_resident_compact_impl = offload.lru_resident_compact
compute_lru_resident_addrs_impl = offload.compute_lru_resident_addrs
get_sparse_kv_plan_workspace_size_impl = offload.get_sparse_kv_plan_workspace_size
sparse_kv_load_runtime_impl = offload.sparse_kv_load_runtime


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


def get_device_address(tensor):
    if tensor.device.type != "cpu":
        raise ValueError(f"offload host-pool tensor must be on CPU, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError("offload host-pool tensor must be contiguous")
    size_bytes = tensor.numel() * tensor.element_size()
    if size_bytes <= 0:
        raise ValueError("offload host-pool tensor must not be empty")
    address = get_device_address_impl(tensor.data_ptr(), size_bytes)
    if address == 0:
        raise RuntimeError(
            "MemFabric host-pool address is not registered for device access"
        )
    return address


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


def get_sparse_kv_plan_workspace_size(num_reqs, topk, capacity):
    """Return descriptor-free Plan workspace bytes (independent of max_token)."""
    return get_sparse_kv_plan_workspace_size_impl(num_reqs, topk, capacity)


def _require_npu_tensor(name, tensor, dtype, shape):
    if tensor.device.type != "npu":
        raise ValueError(f"{name} must be an NPU tensor, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def sparse_kv_load_runtime(
        req_ids, last_req_ids, topk_indices, stable_prefix_lens,
        slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
        miss_slots, compact_workspace, block_table, host_k_base, host_v_base,
        device_k_base, device_v_base, block_size, token_size_bytes_k,
        token_size_bytes_v, max_token, deviceId):
    """Submit runtime Sparse KV Plan + registered-DVA H2D Transfer.

    This is one logical asynchronous API.  It does not allocate or publish the
    legacy gvas/addr/size descriptor buffers.  ``host_*_base`` must be DVA
    values returned by :func:`get_device_address`; ``device_*_base`` are NPU
    tensor data pointers.  The caller owns all persistent resident/LRU state.
    """
    if req_ids.ndim != 1:
        raise ValueError("req_ids must be rank 1")
    if topk_indices.ndim != 2 or slot_to_token.ndim != 2 or block_table.ndim != 2:
        raise ValueError(
            "topk_indices, slot_to_token and block_table must be rank 2"
        )
    num_reqs = req_ids.shape[0]
    topk = topk_indices.shape[1]
    capacity = slot_to_token.shape[1]
    max_num_blocks = block_table.shape[1]
    if num_reqs <= 0 or topk <= 0 or capacity <= 0 or max_num_blocks <= 0:
        raise ValueError("runtime Sparse KV dimensions must all be positive")
    int32_max = (1 << 31) - 1
    if topk > int32_max // 2 or capacity > int32_max:
        raise ValueError("topk/capacity exceed the current int32 Plan ABI")
    if max_token <= 0 or max_token > int32_max:
        raise ValueError("max_token must be representable by int32 token IDs")

    _require_npu_tensor("req_ids", req_ids, torch.int64, (num_reqs,))
    _require_npu_tensor(
        "last_req_ids", last_req_ids, torch.int64, (num_reqs,)
    )
    _require_npu_tensor(
        "stable_prefix_lens", stable_prefix_lens, torch.int32,
        (num_reqs,)
    )
    _require_npu_tensor(
        "topk_indices", topk_indices, torch.int32, (num_reqs, topk)
    )
    _require_npu_tensor(
        "slot_to_token", slot_to_token, torch.int32,
        (num_reqs, capacity)
    )
    _require_npu_tensor(
        "lru_slots", lru_slots, torch.int32, (num_reqs, capacity)
    )
    for name, tensor in (
            ("current_slots", current_slots),
            ("miss_tokens", miss_tokens),
            ("miss_slots", miss_slots)):
        _require_npu_tensor(name, tensor, torch.int32, (num_reqs, topk))
    _require_npu_tensor(
        "miss_count", miss_count, torch.int32, (num_reqs,)
    )
    _require_npu_tensor(
        "block_table", block_table, torch.int32,
        (num_reqs, max_num_blocks)
    )
    runtime_tensors = (
        last_req_ids, topk_indices, stable_prefix_lens, slot_to_token,
        lru_slots, current_slots, miss_count, miss_tokens, miss_slots,
        block_table, compact_workspace,
    )
    if any(tensor.device != req_ids.device for tensor in runtime_tensors):
        raise ValueError("all SparseKvLoadRuntime tensors must share one NPU device")
    if torch.device(deviceId) != req_ids.device:
        raise ValueError(
            f"deviceId {deviceId} does not match tensor device {req_ids.device}"
        )
    if compact_workspace.device.type != "npu" or not compact_workspace.is_contiguous():
        raise ValueError("compact_workspace must be a contiguous NPU tensor")
    workspace_bytes = compact_workspace.numel() * compact_workspace.element_size()
    required_bytes = get_sparse_kv_plan_workspace_size(
        num_reqs, topk, capacity
    )
    if required_bytes <= 0 or workspace_bytes < required_bytes:
        raise ValueError(
            "compact_workspace is too small: "
            f"required={required_bytes}, actual={workspace_bytes}"
        )
    if not all(value > 0 for value in (
            host_k_base, host_v_base, device_k_base, device_v_base,
            block_size, token_size_bytes_k, token_size_bytes_v, max_token)):
        raise ValueError("bases, geometry and max_token must be positive")

    return sparse_kv_load_runtime_impl(
        req_ids.data_ptr(), last_req_ids.data_ptr(),
        topk_indices.data_ptr(), stable_prefix_lens.data_ptr(),
        slot_to_token.data_ptr(), lru_slots.data_ptr(),
        current_slots.data_ptr(), miss_count.data_ptr(),
        miss_tokens.data_ptr(), miss_slots.data_ptr(),
        compact_workspace.data_ptr(), workspace_bytes,
        block_table.data_ptr(), host_k_base, host_v_base, device_k_base,
        device_v_base, num_reqs, topk, capacity, max_token,
        max_num_blocks, block_size, token_size_bytes_k,
        token_size_bytes_v, deviceId.index)
