#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""CPU oracle and source contracts for the all-NPU FSA Plan path.

No CANN/NPU dependency is required.  Device parity remains an explicit A5
validation step.
"""

import copy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
READY = 0x5A4D
EXTERNAL_READY = 0x5A45
DIRECT_LAYOUT = 0x5A44
PAIRED_COPY = 0x5A56


def plan_stride(topk):
    return ((topk + 15) // 16) * 16 + 16


def map_rows(req_ids, last_req_ids):
    logical_to_physical = [-1] * len(req_ids)
    used = [0] * len(last_req_ids)
    for logical, req_id in enumerate(req_ids):
        for physical, old_req_id in enumerate(last_req_ids):
            if not used[physical] and old_req_id == req_id:
                logical_to_physical[logical] = physical
                used[physical] = 1
                break
    for logical in range(len(req_ids)):
        if logical_to_physical[logical] >= 0:
            continue
        selected = next((i for i, old in enumerate(last_req_ids)
                         if not used[i] and old == -1), -1)
        if selected < 0:
            selected = next(i for i, value in enumerate(used) if not value)
        logical_to_physical[logical] = selected
        used[selected] = 1
    return logical_to_physical


def fsa_plan_oracle(case):
    state = copy.deepcopy(case)
    rows = len(case["req_ids"])
    topk = len(case["topk_indices"][0])
    capacity = len(case["slot_to_token"][0])
    resident_capacity = capacity - 1
    max_token = case["max_token"]
    mapping = map_rows(case["req_ids"], state["last_req_ids"])
    current = [[-1] * topk for _ in range(rows)]
    miss_count = [0] * rows
    miss_tokens = [[-1] * topk for _ in range(rows)]
    miss_slots = [[-1] * topk for _ in range(rows)]
    encoded = [[0] * plan_stride(topk) for _ in range(rows)]
    current_linear = [-1] * rows

    for logical, physical in enumerate(mapping):
        slots = state["slot_to_token"][physical]
        lru = state["lru_slots"][physical]
        if state["last_req_ids"][physical] != case["req_ids"][logical]:
            slots[:resident_capacity] = [-1] * resident_capacity
            lru[:resident_capacity] = list(range(resident_capacity))
            state["last_req_ids"][physical] = case["req_ids"][logical]
        stable = min(max(case["stable_prefix_lens"][logical], 0),
                     max_token)
        visible = min(max(case["visible_seq_lens"][logical], 0),
                      max_token)
        first = {}
        for pos, token in enumerate(case["topk_indices"][logical]):
            if 0 <= token < max_token and token < visible:
                first.setdefault(token, pos)
        hit_slots, evictable = [], []
        current_token_slot = resident_capacity
        for order in range(resident_capacity):
            slot = lru[order]
            if not 0 <= slot < resident_capacity:
                continue
            token = slots[slot]
            if 0 <= token < max_token and token >= stable:
                slots[slot] = -1
                token = -1
            if 0 <= token < max_token and token in first:
                pos = first[token]
                current[logical][pos] = slot
                encoded[logical][pos] = slot + 1
                if token == visible - 1:
                    current_token_slot = slot
                hit_slots.append(slot)
            else:
                evictable.append(slot)
        misses = []
        for pos, token in enumerate(case["topk_indices"][logical]):
            if (0 <= token < max_token and token < visible
                    and current[logical][pos] < 0):
                misses.append((token, pos))
        assigned = min(len(misses), len(evictable))
        for index in range(assigned):
            token, pos = misses[index]
            slot = evictable[index]
            slots[slot] = token
            current[logical][pos] = slot
            miss_tokens[logical][index] = token
            miss_slots[logical][index] = slot
            is_current = token == visible - 1
            encoded[logical][pos] = slot + 1 if is_current else -(slot + 1)
            if is_current:
                current_token_slot = slot
        miss_count[logical] = assigned
        rebuilt = evictable[assigned:] + evictable[:assigned] + hit_slots
        lru[:len(rebuilt)] = rebuilt
        current_linear[logical] = physical * capacity + current_token_slot
        control = ((topk + 15) // 16) * 16
        encoded[logical][control:control + 8] = [
            READY, EXTERNAL_READY, topk, 0, physical,
            DIRECT_LAYOUT, capacity, PAIRED_COPY]

    state.update({
        "logical_to_physical": mapping,
        "current_slots": current,
        "miss_count": miss_count,
        "miss_tokens": miss_tokens,
        "miss_slots": miss_slots,
        "encoded_plan": encoded,
        "current_linear_slots": current_linear,
    })
    return state


class FsaPlanOracleTest(unittest.TestCase):
    def test_stable_mapping_current_token_and_unassigned_tail(self):
        case = {
            "req_ids": [22, 11],
            "last_req_ids": [11, 22, -1],
            "topk_indices": [[7, 9, 7, 11], [3, 8, -1, 12]],
            "stable_prefix_lens": [20, 20],
            "visible_seq_lens": [8, 9],
            "max_token": 20,
            "slot_to_token": [[3, -1, -1, 99],
                              [9, -1, -1, 99],
                              [-1, -1, -1, 99]],
            "lru_slots": [[1, 2, 0, 3], [1, 2, 0, 3],
                          [0, 1, 2, 3]],
        }
        result = fsa_plan_oracle(case)
        self.assertEqual(result["logical_to_physical"], [1, 0])
        self.assertEqual(result["miss_count"], [2, 1])
        self.assertEqual(result["miss_tokens"][0][2:], [-1, -1])
        self.assertEqual(result["miss_slots"][0][2:], [-1, -1])
        # token 7 is the current token and appears twice; stable assignment
        # order makes the later duplicate's slot the final injection target.
        self.assertEqual(result["current_linear_slots"][0], 6)
        self.assertGreater(result["encoded_plan"][0][0], 0)
        self.assertGreater(result["encoded_plan"][0][2], 0)

    def test_plan_abi_constants_and_shape(self):
        self.assertEqual(plan_stride(2048), 2064)
        self.assertEqual(plan_stride(2048) * 2, 4128)
        header = (ROOT / "src/acc_offload/include/host/"
                  "acc_offload_sparse_kv_runtime.h").read_text()
        for marker in ("0x5A4D", "0x5A45", "0x5A44", "0x5A56"):
            self.assertIn(marker, header)

    def test_source_contract_is_plan_only_and_same_stream(self):
        launch = (ROOT / "src/acc_offload/csrc/launch/"
                  "acc_offload_operators_launch.cpp").read_text()
        begin = launch.index("void AccOffloadSparseKvPlanFsaRuntime")
        body = launch[begin:]
        self.assertIn("OffloadOpsSparseKvPlanFsaRuntime", body)
        self.assertNotIn("OffloadOpsSparseKvTransferRuntime", body)
        self.assertNotIn("Synchronize", body)
        kernel = (ROOT / "src/acc_offload/csrc/operators/"
                  "acc_offload_sparse_kv_plan_runtime.cpp").read_text()
        self.assertIn("SparseKvMapFsaRowsRuntimeKernel<<<1, 0, stream>>>",
                      kernel)
        self.assertIn("SparseKvPlanFsaRuntimeKernel<<<blockDim", kernel)
        self.assertIn("__gm__ int16_t *encodedPlan", kernel)


if __name__ == "__main__":
    unittest.main()
