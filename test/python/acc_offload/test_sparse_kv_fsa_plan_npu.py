#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Opt-in A5 parity test for the all-NPU FSA Plan-only path.

Run after installing the matching production package:

    MF_RUN_A5_FSA_PLAN_RUNTIME=1 python3 \
      test/python/acc_offload/test_sparse_kv_fsa_plan_npu.py -v

The test synchronizes only after MapRows and PlanFsa have been submitted.  It
does not exercise TransferRuntime; fused-attention integration is validated
by the companion vLLM branch.
"""

import copy
import os
import sys
import unittest
from pathlib import Path


RUN_A5 = os.getenv("MF_RUN_A5_FSA_PLAN_RUNTIME") == "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sparse_kv_fsa_plan import fsa_plan_oracle  # noqa: E402

if RUN_A5:
    import torch
    import torch_npu  # noqa: F401
    from memfabric_hybrid import offload


@unittest.skipUnless(RUN_A5, "set MF_RUN_A5_FSA_PLAN_RUNTIME=1 on A5")
class TestSparseKvFsaPlanA5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        config = offload.OffloadConfig()
        config.device_id = 0
        config.reserve_size = 1 << 30
        config.alloc_size = 1 << 30
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.SHARED
        config.register_host_memory = True
        if offload.initialize(config) != 0:
            raise RuntimeError("offload.initialize failed")

    @classmethod
    def tearDownClass(cls):
        offload.uninitialize()

    def _npu(self, value, dtype):
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _run_sequence(self, initial_state, logical_steps):
        """Run multiple calls while preserving both CPU and NPU state.

        Only logical inputs change between steps. ``last_req_ids``,
        ``slot_to_token`` and ``lru_slots`` are allocated once and fed back to
        every subsequent kernel invocation. This is the property required to
        catch a broken stable physical-row implementation.
        """
        physical = len(initial_state["last_req_ids"])
        topk = len(logical_steps[0]["topk_indices"][0])
        capacity = len(initial_state["slot_to_token"][0])
        max_token = initial_state["max_token"]

        last_req_ids = self._npu(
            initial_state["last_req_ids"], torch.int64)
        slot_to_token = self._npu(
            initial_state["slot_to_token"], torch.int32)
        lru_slots = self._npu(initial_state["lru_slots"], torch.int32)
        cpu_state = copy.deepcopy(initial_state)

        for step_index, logical_inputs in enumerate(logical_steps):
            logical = len(logical_inputs["req_ids"])
            self.assertLessEqual(logical, physical)
            self.assertTrue(all(
                len(row) == topk
                for row in logical_inputs["topk_indices"]
            ))
            oracle_case = {
                **copy.deepcopy(logical_inputs),
                "last_req_ids": copy.deepcopy(cpu_state["last_req_ids"]),
                "slot_to_token": copy.deepcopy(cpu_state["slot_to_token"]),
                "lru_slots": copy.deepcopy(cpu_state["lru_slots"]),
                "max_token": max_token,
            }
            expected = fsa_plan_oracle(oracle_case)

            req_ids = self._npu(logical_inputs["req_ids"], torch.int64)
            topk_indices = self._npu(
                logical_inputs["topk_indices"], torch.int32)
            stable = self._npu(
                logical_inputs["stable_prefix_lens"], torch.int32)
            visible = self._npu(
                logical_inputs["visible_seq_lens"], torch.int32)
            current = torch.full(
                (logical, topk), -99, dtype=torch.int32,
                device=self.device)
            miss_count = torch.full(
                (logical,), -99, dtype=torch.int32, device=self.device)
            miss_tokens = torch.full(
                (logical, topk), -99, dtype=torch.int32,
                device=self.device)
            miss_slots = torch.full(
                (logical, topk), -99, dtype=torch.int32,
                device=self.device)
            compact_bytes = offload.get_sparse_kv_plan_workspace_size(
                logical, topk, capacity)
            row_map_bytes = offload.get_sparse_kv_fsa_row_map_workspace_size(
                logical, physical)
            compact = torch.empty(
                compact_bytes, dtype=torch.uint8, device=self.device)
            row_map = torch.empty(
                row_map_bytes, dtype=torch.uint8, device=self.device)
            stride = offload.get_sparse_kv_fsa_plan_row_stride(topk)
            encoded = torch.full(
                (logical, stride), -99, dtype=torch.int16,
                device=self.device)
            current_linear = torch.full(
                (logical,), -99, dtype=torch.int32, device=self.device)

            result = offload.sparse_kv_plan_fsa_runtime(
                req_ids, last_req_ids, topk_indices, stable, visible,
                slot_to_token, lru_slots, current, miss_count, miss_tokens,
                miss_slots, compact, row_map, encoded, current_linear,
                max_token, self.device)
            self.assertEqual(result, 0, f"step={step_index}")
            torch.npu.synchronize()

            mapping = row_map.view(torch.int32)[:logical].cpu().tolist()
            self.assertEqual(
                mapping, expected["logical_to_physical"],
                f"mapping step={step_index}")
            self.assertEqual(last_req_ids.cpu().tolist(),
                             expected["last_req_ids"])
            self.assertEqual(slot_to_token.cpu().tolist(),
                             expected["slot_to_token"])
            self.assertEqual(lru_slots.cpu().tolist(),
                             expected["lru_slots"])
            self.assertEqual(current.cpu().tolist(),
                             expected["current_slots"])
            self.assertEqual(miss_count.cpu().tolist(),
                             expected["miss_count"])
            self.assertEqual(miss_tokens.cpu().tolist(),
                             expected["miss_tokens"])
            self.assertEqual(miss_slots.cpu().tolist(),
                             expected["miss_slots"])
            self.assertEqual(encoded.cpu().tolist(),
                             expected["encoded_plan"])
            self.assertEqual(current_linear.cpu().tolist(),
                             expected["current_linear_slots"])

            cpu_state = {
                "last_req_ids": copy.deepcopy(expected["last_req_ids"]),
                "slot_to_token": copy.deepcopy(expected["slot_to_token"]),
                "lru_slots": copy.deepcopy(expected["lru_slots"]),
                "max_token": max_token,
            }

    @staticmethod
    def _empty_state(physical_rows, capacity, max_token=64):
        return {
            "last_req_ids": [-1] * physical_rows,
            "slot_to_token": [[-1] * capacity
                              for _ in range(physical_rows)],
            "lru_slots": [list(range(capacity))
                          for _ in range(physical_rows)],
            "max_token": max_token,
        }

    def test_persistent_reorder_and_replacement(self):
        initial = self._empty_state(2, 5)
        steps = [
            {
                "req_ids": [11, 22],
                "topk_indices": [[1, 2, 3, 4, 5, 6],
                                 [7, 8, 9, 10, 11, 12]],
                "stable_prefix_lens": [64, 64],
                "visible_seq_lens": [7, 13],
            },
            {
                # This must map to physical rows [1, 0] from step 0. The NPU
                # last_req_ids/slot/LRU tensors are deliberately not reset.
                "req_ids": [22, 11],
                "topk_indices": [[7, 9, 12, 13, 14, 8],
                                 [1, 3, 6, 15, 16, 2]],
                "stable_prefix_lens": [12, 6],
                "visible_seq_lens": [15, 17],
            },
            {
                # All physical rows are occupied. req=22 keeps row 1 and the
                # replacement req=33 deterministically receives row 0.
                "req_ids": [33, 22],
                "topk_indices": [[20, 21, 22, 23, 24, 25],
                                 [7, 8, 9, 10, 11, 12]],
                "stable_prefix_lens": [64, 64],
                "visible_seq_lens": [26, 13],
            },
        ]
        self._run_sequence(initial, steps)

    def test_rows_1_2_4_and_semantic_boundaries(self):
        cases = [
            (1, 4, [{
                "req_ids": [1],
                # Five valid misses compete for three resident slots.
                "topk_indices": [[2, 3, 4, 5, 6, -1]],
                "stable_prefix_lens": [64],
                "visible_seq_lens": [7],
            }]),
            (2, 5, [{
                "req_ids": [2, 3],
                # Row 0 includes duplicate current-token misses; row 1 has
                # invisible/invalid tokens and current token not in TopK.
                "topk_indices": [[7, 9, 7, 11, 5, 6],
                                 [3, 20, -1, 12, 4, 5]],
                "stable_prefix_lens": [64, 4],
                "visible_seq_lens": [8, 9],
            }]),
            (4, 6, [{
                "req_ids": [4, 5, 6, 7],
                "topk_indices": [[1, 2, 3, 4, 5, 6],
                                 [8, 9, 10, 11, 12, 13],
                                 [14, 14, 15, 16, 17, 18],
                                 [-1, 63, 19, 20, 21, 22]],
                "stable_prefix_lens": [64, 10, 16, 20],
                "visible_seq_lens": [7, 14, 19, 23],
            }]),
        ]
        for rows, capacity, steps in cases:
            with self.subTest(rows=rows, capacity=capacity):
                self._run_sequence(
                    self._empty_state(rows, capacity), steps)

    def test_existing_hits_and_current_token_hit(self):
        initial = {
            "last_req_ids": [11, 22, -1, -1],
            "slot_to_token": [[3, -1, -1, -1, 99],
                              [7, 9, -1, -1, 99],
                              [-1, -1, -1, -1, 99],
                              [-1, -1, -1, -1, 99]],
            "lru_slots": [[1, 2, 3, 0, 4], [2, 3, 1, 0, 4],
                          [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            "max_token": 20,
        }
        steps = [{
            "req_ids": [22, 11],
            "topk_indices": [[7, 9, 7, 11, 5, 6],
                             [3, 8, -1, 12, 4, 5]],
            "stable_prefix_lens": [20, 20],
            "visible_seq_lens": [8, 9],
        }]
        self._run_sequence(initial, steps)


if __name__ == "__main__":
    unittest.main()
