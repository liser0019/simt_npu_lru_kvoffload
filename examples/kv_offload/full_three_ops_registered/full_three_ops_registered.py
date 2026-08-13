#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""Single-rank Compact -> Addrs -> registered-Host SparseCopy test."""

import argparse

import torch
import torch_npu
import memfabric_hybrid as mf
from memfabric_hybrid import offload


ONE_GIB = 1 << 30
WORKSPACE_THREADS = 8
NUM_REQS = 1
TOPK = 2
CAPACITY = 4
MAX_TOKEN = 16
BLOCK_SIZE = 4
K_BYTES = 32
V_BYTES = 16


def _initialize(device_id: int) -> None:
    config = offload.OffloadConfig()
    config.device_id = device_id
    config.reserve_size = ONE_GIB
    config.alloc_size = ONE_GIB
    config.world_size = 1
    config.rank_id = 0
    config.scene = offload.Scene.SHARED
    config.register_host_memory = True
    if offload.initialize(config) != 0:
        raise RuntimeError("offload.initialize failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    mf.set_log_level(3)
    _initialize(args.device)
    device = torch.device("npu", args.device)

    try:
        host_k = offload.empty([MAX_TOKEN, K_BYTES], dtype=torch.uint8)
        host_v = offload.empty([MAX_TOKEN, V_BYTES], dtype=torch.uint8)
        for token in range(MAX_TOKEN):
            host_k[token].fill_((token * 7 + 1) % 251)
            host_v[token].fill_((token * 11 + 3) % 251)
        host_k_device = int(offload.get_device_address(host_k))
        host_v_device = int(offload.get_device_address(host_v))

        resident_k = torch.zeros([CAPACITY, K_BYTES], dtype=torch.uint8, device=device)
        resident_v = torch.zeros([CAPACITY, V_BYTES], dtype=torch.uint8, device=device)

        req_ids = torch.tensor([42], dtype=torch.int64, device=device)
        last_req_ids = torch.full([NUM_REQS], -1, dtype=torch.int64, device=device)
        topk_indices = torch.tensor([[1, 6]], dtype=torch.int32, device=device)
        stable_prefix_lens = torch.tensor([MAX_TOKEN], dtype=torch.int32, device=device)
        slot_to_token = torch.full(
            [NUM_REQS, CAPACITY], -1, dtype=torch.int32, device=device
        )
        lru_slots = torch.arange(CAPACITY, dtype=torch.int32, device=device).view(
            NUM_REQS, CAPACITY
        )
        current_slots = torch.empty(
            [NUM_REQS, TOPK], dtype=torch.int32, device=device
        )
        miss_count = torch.empty([NUM_REQS], dtype=torch.int32, device=device)
        miss_tokens = torch.empty(
            [NUM_REQS, TOPK], dtype=torch.int32, device=device
        )
        miss_slots = torch.empty(
            [NUM_REQS, TOPK], dtype=torch.int32, device=device
        )
        token_mark = torch.zeros(
            [WORKSPACE_THREADS, MAX_TOKEN], dtype=torch.int32, device=device
        )
        token_pos = torch.full(
            [WORKSPACE_THREADS, MAX_TOKEN], -1, dtype=torch.int32, device=device
        )
        epochs = torch.zeros([WORKSPACE_THREADS], dtype=torch.int32, device=device)

        result = offload.lru_resident_compact(
            req_ids,
            last_req_ids,
            topk_indices,
            stable_prefix_lens,
            slot_to_token,
            lru_slots,
            current_slots,
            miss_count,
            miss_tokens,
            miss_slots,
            token_mark,
            token_pos,
            epochs,
            NUM_REQS,
            TOPK,
            CAPACITY,
            MAX_TOKEN,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"lru_resident_compact returned {result}")

        block_table = torch.arange(
            MAX_TOKEN // BLOCK_SIZE, dtype=torch.int32, device=device
        ).view(NUM_REQS, -1)
        descriptor_capacity = 2 * NUM_REQS * TOPK
        sources = torch.zeros(descriptor_capacity, dtype=torch.int64, device=device)
        destinations = torch.zeros(
            descriptor_capacity, dtype=torch.int64, device=device
        )
        lengths = torch.zeros(descriptor_capacity, dtype=torch.int32, device=device)
        count = torch.zeros(1, dtype=torch.int32, device=device)

        result = offload.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            sources,
            destinations,
            lengths,
            count,
            BLOCK_SIZE,
            K_BYTES,
            V_BYTES,
            host_k_device,
            host_v_device,
            resident_k.data_ptr(),
            resident_v.data_ptr(),
            CAPACITY,
            NUM_REQS,
            TOPK,
            block_table.shape[1],
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"compute_lru_resident_addrs returned {result}")

        result = offload.sparse_copy(
            sources,
            destinations,
            lengths,
            count,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"sparse_copy returned {result}")
        torch.npu.synchronize()

        expected_tokens = [1, 6]
        actual_slots = current_slots.cpu().reshape(-1).tolist()
        if miss_count.cpu().tolist() != [TOPK] or actual_slots != [0, 1]:
            raise AssertionError(
                "unexpected LRU result: "
                f"miss_count={miss_count.cpu().tolist()} slots={actual_slots}"
            )
        for token, slot in zip(expected_tokens, actual_slots, strict=True):
            if not torch.equal(resident_k[slot].cpu(), host_k[token]):
                raise AssertionError(f"K mismatch for token={token}, slot={slot}")
            if not torch.equal(resident_v[slot].cpu(), host_v[token]):
                raise AssertionError(f"V mismatch for token={token}, slot={slot}")

        print(
            "[PASS] lru_resident_compact -> compute_lru_resident_addrs "
            "-> registered-host sparse_copy",
            flush=True,
        )
        print("FULL_THREE_OPS_REGISTERED_PASS", flush=True)
    finally:
        offload.uninitialize()


if __name__ == "__main__":
    main()
