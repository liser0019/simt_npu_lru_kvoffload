#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""CPU oracle and source-contract tests for SparseKvLoadRuntime.

These tests intentionally require neither torch_npu nor CANN.  A5 execution is
validated separately after the mixed/AIV sources have been compiled by
Bisheng.  The CPU Plan oracle mirrors the vLLM-Ascend implementation, including
its non-obvious duplicate rule: only the first TopK occurrence can be a hit;
later duplicate positions remain independent misses.
"""

import copy
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def workspace_layout(num_reqs, topk, capacity):
    if num_reqs <= 0 or topk <= 0 or capacity <= 0:
        return 0, 0, 0
    target = 2 * topk
    hash_capacity = 32
    while hash_capacity < target:
        hash_capacity <<= 1
    row_elements = 3 * hash_capacity + 2 * capacity + topk
    return hash_capacity, row_elements, 4 * num_reqs * row_elements


def plan_oracle(case):
    """Mutate a copy of resident state and return the exact Plan outputs."""
    state = copy.deepcopy(case)
    rows = case["num_reqs"]
    topk = case["topk"]
    capacity = case["capacity"]
    max_token = case["max_token"]
    current_slots = [[-1] * topk for _ in range(rows)]
    miss_count = [0] * rows
    miss_tokens = [[-1] * topk for _ in range(rows)]
    miss_slots = [[-1] * topk for _ in range(rows)]

    for row in range(rows):
        slot_to_token = state["slot_to_token"][row]
        lru = state["lru_slots"][row]
        if state["last_req_ids"][row] != state["req_ids"][row]:
            slot_to_token[:] = [-1] * capacity
            lru[:] = list(range(capacity))
            state["last_req_ids"][row] = state["req_ids"][row]

        stable_prefix = min(max(state["stable_prefix_lens"][row], 0),
                            max_token)
        first_position = {}
        for position, token in enumerate(state["topk_indices"][row]):
            if 0 <= token < max_token and token not in first_position:
                first_position[token] = position

        hit_slots = []
        evictable_slots = []
        for slot in lru:
            if slot < 0 or slot >= capacity:
                continue
            token = slot_to_token[slot]
            if 0 <= token < max_token and token >= stable_prefix:
                slot_to_token[slot] = -1
                token = -1
            if 0 <= token < max_token and token in first_position:
                current_slots[row][first_position[token]] = slot
                hit_slots.append(slot)
            else:
                evictable_slots.append(slot)

        misses = []
        for position, token in enumerate(state["topk_indices"][row]):
            if 0 <= token < max_token and current_slots[row][position] < 0:
                misses.append((token, position))

        assign_count = min(len(misses), len(evictable_slots))
        assigned_slots = []
        for miss_index in range(assign_count):
            token, position = misses[miss_index]
            slot = evictable_slots[miss_index]
            slot_to_token[slot] = token
            current_slots[row][position] = slot
            miss_tokens[row][miss_index] = token
            miss_slots[row][miss_index] = slot
            assigned_slots.append(slot)
        miss_count[row] = assign_count

        rebuilt = (evictable_slots[assign_count:] + assigned_slots
                   + hit_slots)
        lru[:len(rebuilt)] = rebuilt

    state.update({
        "current_slots": current_slots,
        "miss_count": miss_count,
        "miss_tokens": miss_tokens,
        "miss_slots": miss_slots,
    })
    return state


def transfer_oracle(case, plan, host_k, host_v, device_k, device_v):
    """Execute descriptor-free token/slot -> address -> payload copy on CPU."""
    block_size = case["block_size"]
    bytes_k = case["token_size_bytes_k"]
    bytes_v = case["token_size_bytes_v"]
    max_blocks = case["max_num_blocks"]
    capacity = case["capacity"]
    topk = case["topk"]
    for row in range(case["num_reqs"]):
        count = min(max(plan["miss_count"][row], 0), topk)
        for index in range(count):
            token = plan["miss_tokens"][row][index]
            slot = plan["miss_slots"][row][index]
            if token < 0 or slot < 0 or slot >= capacity:
                continue
            block_id = token // block_size
            if block_id < 0 or block_id >= max_blocks:
                continue
            block_index = case["block_table"][row][block_id]
            if block_index < 0:
                continue
            offset = token % block_size
            source_k = (block_index * block_size + offset) * bytes_k
            source_v = (block_index * block_size + offset) * bytes_v
            linear_slot = row * capacity + slot
            destination_k = linear_slot * bytes_k
            destination_v = linear_slot * bytes_v
            device_k[destination_k:destination_k + bytes_k] = \
                host_k[source_k:source_k + bytes_k]
            device_v[destination_v:destination_v + bytes_v] = \
                host_v[source_v:source_v + bytes_v]
    return device_k, device_v


def make_case(num_reqs=3, topk=33, capacity=47, max_token=128):
    max_num_blocks = (max_token + 3) // 4
    topk_rows = []
    slot_rows = []
    lru_rows = []
    block_table = []
    for row in range(num_reqs):
        tokens = [(row * 7 + position * 3) % max_token
                  for position in range(topk)]
        if topk >= 4:
            tokens[1] = tokens[0]  # later duplicate is an independent miss
            tokens[2] = -1
            tokens[3] = max_token
        topk_rows.append(tokens)
        resident = [-1] * capacity
        for slot in range(min(capacity, max(1, topk // 3))):
            resident[slot] = (row * 11 + slot * 5) % max_token
        slot_rows.append(resident)
        rng = random.Random(20260814 + row)
        order = list(range(capacity))
        rng.shuffle(order)
        lru_rows.append(order)
        block_table.append([
            row * max_num_blocks + block for block in range(max_num_blocks)
        ])
    return {
        "num_reqs": num_reqs,
        "topk": topk,
        "capacity": capacity,
        "max_token": max_token,
        "req_ids": list(range(100, 100 + num_reqs)),
        "last_req_ids": list(range(100, 100 + num_reqs)),
        "stable_prefix_lens": [max_token] * num_reqs,
        "topk_indices": topk_rows,
        "slot_to_token": slot_rows,
        "lru_slots": lru_rows,
        "block_size": 4,
        "token_size_bytes_k": 8,
        "token_size_bytes_v": 3,
        "max_num_blocks": max_num_blocks,
        "block_table": block_table,
    }


class TestSparseKvRuntimeCpuOracle(unittest.TestCase):
    def test_duplicate_semantics_match_cpu_contract(self):
        case = make_case(1, 4, 4, 16)
        case["topk_indices"] = [[5, 5, 7, -1]]
        case["slot_to_token"] = [[5, -1, -1, -1]]
        case["lru_slots"] = [[1, 2, 3, 0]]
        plan = plan_oracle(case)
        self.assertEqual(plan["current_slots"][0][0], 0)
        self.assertGreaterEqual(plan["current_slots"][0][1], 0)
        self.assertNotEqual(plan["current_slots"][0][0],
                            plan["current_slots"][0][1])
        self.assertIn(5, plan["miss_tokens"][0][:plan["miss_count"][0]])

    def test_request_reset_and_stable_prefix(self):
        case = make_case(2, 8, 12, 64)
        case["last_req_ids"][0] = -1
        case["stable_prefix_lens"][1] = 10
        case["slot_to_token"][1][0] = 12
        plan = plan_oracle(case)
        self.assertEqual(plan["last_req_ids"][0], case["req_ids"][0])
        self.assertTrue(all(0 <= slot < case["capacity"]
                            for slot in plan["lru_slots"][0]))
        self.assertNotEqual(plan["slot_to_token"][1][0], 12)

    def test_runtime_shape_matrix(self):
        matrix = [
            (1, 1, 1, 16), (2, 31, 47, 128),
            (2, 33, 7, 128),
            (3, 32, 64, 4096), (4, 33, 97, 8192),
            (7, 511, 777, 32768), (8, 1024, 1537, 131072),
        ]
        for shape in matrix:
            with self.subTest(shape=shape):
                plan = plan_oracle(make_case(*shape))
                self.assertEqual(len(plan["miss_count"]), shape[0])
                for row, count in enumerate(plan["miss_count"]):
                    self.assertGreaterEqual(count, 0)
                    self.assertLessEqual(count, min(shape[1], shape[2]))
                    self.assertTrue(all(
                        0 <= slot < shape[2]
                        for slot in plan["current_slots"][row]
                        if slot >= 0
                    ))

    def test_unassigned_miss_tail_remains_invalid(self):
        case = make_case(2, 33, 7, 128)
        plan = plan_oracle(case)
        for row, count in enumerate(plan["miss_count"]):
            self.assertLessEqual(count, case["capacity"])
            self.assertEqual(
                plan["miss_tokens"][row][count:],
                [-1] * (case["topk"] - count),
            )
            self.assertEqual(
                plan["miss_slots"][row][count:],
                [-1] * (case["topk"] - count),
            )

    def test_random_properties(self):
        rng = random.Random(20260814)
        for _ in range(50):
            num_reqs = rng.randint(1, 5)
            topk = rng.randint(1, 80)
            capacity = rng.randint(1, 96)
            max_token = rng.randint(max(topk, 8), 256)
            case = make_case(num_reqs, topk, capacity, max_token)
            for row in range(num_reqs):
                case["stable_prefix_lens"][row] = rng.randint(-5,
                                                               max_token + 5)
                if rng.random() < 0.3:
                    case["last_req_ids"][row] = -1
            plan = plan_oracle(case)
            for row in range(num_reqs):
                count = plan["miss_count"][row]
                self.assertEqual(
                    plan["miss_tokens"][row][count:], [-1] * (topk - count)
                )
                self.assertEqual(
                    plan["miss_slots"][row][count:], [-1] * (topk - count)
                )

    def test_descriptor_free_transfer_payload_and_request_stride(self):
        case = make_case(3, 33, 47, 128)
        plan = plan_oracle(case)
        total_blocks = case["num_reqs"] * case["max_num_blocks"]
        host_k = bytearray((index * 13 + 1) & 0xFF for index in range(
            total_blocks * case["block_size"] * case["token_size_bytes_k"]
        ))
        host_v = bytearray((index * 17 + 9) & 0xFF for index in range(
            total_blocks * case["block_size"] * case["token_size_bytes_v"]
        ))
        device_k = bytearray([0xA5] * (
            case["num_reqs"] * case["capacity"]
            * case["token_size_bytes_k"]
        ))
        device_v = bytearray([0x5A] * (
            case["num_reqs"] * case["capacity"]
            * case["token_size_bytes_v"]
        ))
        transfer_oracle(case, plan, host_k, host_v, device_k, device_v)
        for row in range(case["num_reqs"]):
            for index in range(plan["miss_count"][row]):
                token = plan["miss_tokens"][row][index]
                slot = plan["miss_slots"][row][index]
                block = case["block_table"][row][token // case["block_size"]]
                source = (block * case["block_size"]
                          + token % case["block_size"])
                dst = row * case["capacity"] + slot
                kb = case["token_size_bytes_k"]
                vb = case["token_size_bytes_v"]
                self.assertEqual(device_k[dst * kb:(dst + 1) * kb],
                                 host_k[source * kb:(source + 1) * kb])
                self.assertEqual(device_v[dst * vb:(dst + 1) * vb],
                                 host_v[source * vb:(source + 1) * vb])

    def test_workspace_does_not_scale_with_max_token(self):
        values = [workspace_layout(8, 2048, 4096)[2]
                  for _max_token in (4096, 131072, 1048576)]
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])
        hash_capacity, row_elements, size = workspace_layout(8, 2048, 4096)
        self.assertEqual(hash_capacity, 4096)
        self.assertEqual(row_elements, 3 * 4096 + 2 * 4096 + 2048)
        self.assertEqual(size, 8 * row_elements * 4)


class TestSparseKvRuntimeSourceContract(unittest.TestCase):
    def read(self, relative):
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_plan_uses_mixed_gm_hash_and_runtime_tiling(self):
        source = self.read(
            "src/acc_offload/csrc/operators/"
            "acc_offload_sparse_kv_plan_runtime.cpp"
        )
        self.assertIn("__global__ __vector__", source)
        self.assertIn("asc_vf_call<SparseKvPlanRuntimeVf>", source)
        self.assertIn("asc_atomic_cas", source)
        self.assertIn("for (int64_t tile = 0; tile < capacity;", source)
        self.assertIn("for (int64_t tile = 0; tile < topk;", source)
        self.assertNotIn("topk == 2048", source)
        self.assertNotIn("capacity == 4096", source)

    def test_transfer_has_no_descriptor_materialization(self):
        source = self.read(
            "src/acc_offload/csrc/operators/"
            "acc_offload_sparse_kv_transfer_runtime.cpp"
        )
        self.assertIn("TRANSFER_BLOCKS = 32", source)
        self.assertIn("DataCopyPad", source)
        self.assertNotIn("gvas_buffer", source)
        self.assertNotIn("num_tokens_buffer", source)

    def test_one_api_submits_plan_then_transfer_without_sync(self):
        source = self.read(
            "src/acc_offload/csrc/launch/acc_offload_operators_launch.cpp"
        )
        begin = source.index("void AccOffloadSparseKvLoadRuntime")
        body = source[begin:]
        self.assertLess(body.index("OffloadOpsSparseKvPlanRuntime"),
                        body.index("OffloadOpsSparseKvTransferRuntime"))
        self.assertNotIn("Synchronize", body)


if __name__ == "__main__":
    unittest.main()
