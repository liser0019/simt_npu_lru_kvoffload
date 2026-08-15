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

    def _run_case(self, case):
        expected = fsa_plan_oracle(case)
        logical = len(case["req_ids"])
        physical = len(case["last_req_ids"])
        topk = len(case["topk_indices"][0])
        capacity = len(case["slot_to_token"][0])

        req_ids = self._npu(case["req_ids"], torch.int64)
        last_req_ids = self._npu(case["last_req_ids"], torch.int64)
        topk_indices = self._npu(case["topk_indices"], torch.int32)
        stable = self._npu(case["stable_prefix_lens"], torch.int32)
        visible = self._npu(case["visible_seq_lens"], torch.int32)
        slot_to_token = self._npu(case["slot_to_token"], torch.int32)
        lru_slots = self._npu(case["lru_slots"], torch.int32)
        current = torch.full(
            (logical, topk), -99, dtype=torch.int32, device=self.device)
        miss_count = torch.full(
            (logical,), -99, dtype=torch.int32, device=self.device)
        miss_tokens = torch.full(
            (logical, topk), -99, dtype=torch.int32, device=self.device)
        miss_slots = torch.full(
            (logical, topk), -99, dtype=torch.int32, device=self.device)
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
            (logical, stride), -99, dtype=torch.int16, device=self.device)
        current_linear = torch.full(
            (logical,), -99, dtype=torch.int32, device=self.device)

        result = offload.sparse_kv_plan_fsa_runtime(
            req_ids, last_req_ids, topk_indices, stable, visible,
            slot_to_token, lru_slots, current, miss_count, miss_tokens,
            miss_slots, compact, row_map, encoded, current_linear,
            case["max_token"], self.device)
        self.assertEqual(result, 0)
        torch.npu.synchronize()

        mapping = row_map.view(torch.int32)[:logical].cpu().tolist()
        self.assertEqual(mapping, expected["logical_to_physical"])
        self.assertEqual(last_req_ids.cpu().tolist(),
                         expected["last_req_ids"])
        self.assertEqual(slot_to_token.cpu().tolist(),
                         expected["slot_to_token"])
        self.assertEqual(lru_slots.cpu().tolist(), expected["lru_slots"])
        self.assertEqual(current.cpu().tolist(), expected["current_slots"])
        self.assertEqual(miss_count.cpu().tolist(), expected["miss_count"])
        self.assertEqual(miss_tokens.cpu().tolist(), expected["miss_tokens"])
        self.assertEqual(miss_slots.cpu().tolist(), expected["miss_slots"])
        self.assertEqual(encoded.cpu().tolist(), expected["encoded_plan"])
        self.assertEqual(current_linear.cpu().tolist(),
                         expected["current_linear_slots"])

    def test_mapping_visibility_current_token_and_capacity_pressure(self):
        base = {
            "req_ids": [22, 11],
            "last_req_ids": [11, 22, -1, -1],
            "topk_indices": [[7, 9, 7, 11], [3, 8, -1, 12]],
            "stable_prefix_lens": [20, 4],
            "visible_seq_lens": [8, 9],
            "max_token": 20,
            "slot_to_token": [[3, -1, -1, 99],
                              [9, -1, -1, 99],
                              [-1, -1, -1, 99],
                              [-1, -1, -1, 99]],
            "lru_slots": [[1, 2, 0, 3], [1, 2, 0, 3],
                          [0, 1, 2, 3], [0, 1, 2, 3]],
        }
        self._run_case(copy.deepcopy(base))

        # Request reorder on the next invocation must preserve physical rows.
        reordered = copy.deepcopy(base)
        reordered["req_ids"] = [11, 22]
        reordered["topk_indices"].reverse()
        reordered["stable_prefix_lens"].reverse()
        reordered["visible_seq_lens"].reverse()
        self._run_case(reordered)


if __name__ == "__main__":
    unittest.main()
