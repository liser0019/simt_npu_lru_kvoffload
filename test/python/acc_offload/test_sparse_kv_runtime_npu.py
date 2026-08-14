#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""Opt-in A5 integration test for descriptor-free SparseKvLoadRuntime.

Run only after installing the branch-built package on Ascend 950PR/950DT:

    MF_RUN_A5_SPARSE_KV_RUNTIME=1 python3 \
      test/python/acc_offload/test_sparse_kv_runtime_npu.py -v

The logical API queues Plan and Transfer on the same stream.  This test calls
``torch.npu.synchronize()`` only after the API returns, then compares Plan state
against the independent CPU oracle and checks registered-host K/V payloads.
"""

import os
import sys
import unittest
from pathlib import Path


RUN_A5 = os.getenv("MF_RUN_A5_SPARSE_KV_RUNTIME") == "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sparse_kv_runtime import make_case, plan_oracle  # noqa: E402

if RUN_A5:
    import torch
    import torch_npu  # noqa: F401
    from memfabric_hybrid import offload


ONE_GIB = 1 << 30


@unittest.skipUnless(RUN_A5, "set MF_RUN_A5_SPARSE_KV_RUNTIME=1 on A5")
class TestSparseKvRuntimeA5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        config = offload.OffloadConfig()
        config.device_id = 0
        config.reserve_size = ONE_GIB
        config.alloc_size = ONE_GIB
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.SHARED
        config.register_host_memory = True
        if offload.initialize(config) != 0:
            raise RuntimeError("offload.initialize failed")
        manifest_path = os.path.join(
            os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"],
            "acc_offload_kernel_impl.info",
        )
        with open(manifest_path, encoding="utf-8") as manifest_file:
            cls.manifest = dict(
                line.strip().split("=", 1) for line in manifest_file
                if "=" in line
            )
        if cls.manifest.get("sparse_kv_load_runtime") != \
                "plan_plus_transfer":
            raise RuntimeError(f"wrong runtime package: {cls.manifest}")

    @classmethod
    def tearDownClass(cls):
        offload.uninitialize()

    def npu(self, values, dtype):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def run_case(self, num_reqs, topk, capacity, max_token):
        case = make_case(num_reqs, topk, capacity, max_token)
        if capacity >= 4 and topk >= 4:
            # Exercise the non-ideal states that are easy to lose in a
            # parallel rewrite: invalid LRU entries and duplicate resident
            # tokens.  The CPU oracle defines the exact expected behavior.
            case["lru_slots"][0][0] = -1
            case["lru_slots"][0][1] = capacity
            resident_token = case["topk_indices"][0][0]
            case["slot_to_token"][0][2] = resident_token
            case["slot_to_token"][0][3] = resident_token
        expected = plan_oracle(case)
        max_blocks = case["max_num_blocks"]
        block_size = case["block_size"]
        k_bytes = case["token_size_bytes_k"]
        v_bytes = case["token_size_bytes_v"]
        physical_tokens = num_reqs * max_blocks * block_size

        host_k = offload.empty([physical_tokens, k_bytes], dtype=torch.uint8)
        host_v = offload.empty([physical_tokens, v_bytes], dtype=torch.uint8)
        host_k.copy_(torch.tensor([
            [(token * 13 + byte + 1) % 251 for byte in range(k_bytes)]
            for token in range(physical_tokens)
        ], dtype=torch.uint8))
        host_v.copy_(torch.tensor([
            [(token * 17 + byte + 3) % 251 for byte in range(v_bytes)]
            for token in range(physical_tokens)
        ], dtype=torch.uint8))
        host_k_dva = int(offload.get_device_address(host_k))
        host_v_dva = int(offload.get_device_address(host_v))

        req_ids = self.npu(case["req_ids"], torch.int64)
        last_req_ids = self.npu(case["last_req_ids"], torch.int64)
        topk_indices = self.npu(case["topk_indices"], torch.int32)
        stable_prefix = self.npu(case["stable_prefix_lens"], torch.int32)
        slot_to_token = self.npu(case["slot_to_token"], torch.int32)
        lru_slots = self.npu(case["lru_slots"], torch.int32)
        current_slots = torch.full(
            (num_reqs, topk), -99, dtype=torch.int32, device=self.device
        )
        miss_count = torch.full(
            (num_reqs,), -99, dtype=torch.int32, device=self.device
        )
        miss_tokens = torch.full(
            (num_reqs, topk), -99, dtype=torch.int32, device=self.device
        )
        miss_slots = torch.full(
            (num_reqs, topk), -99, dtype=torch.int32, device=self.device
        )
        block_table = self.npu(case["block_table"], torch.int32)
        workspace_bytes = offload.get_sparse_kv_plan_workspace_size(
            num_reqs, topk, capacity
        )
        workspace = torch.empty(
            (workspace_bytes,), dtype=torch.uint8, device=self.device
        )
        resident_k = torch.zeros(
            (num_reqs, capacity, k_bytes), dtype=torch.uint8,
            device=self.device,
        )
        resident_v = torch.zeros(
            (num_reqs, capacity, v_bytes), dtype=torch.uint8,
            device=self.device,
        )

        result = offload.sparse_kv_load_runtime(
            req_ids, last_req_ids, topk_indices, stable_prefix,
            slot_to_token, lru_slots, current_slots, miss_count,
            miss_tokens, miss_slots, workspace, block_table, host_k_dva,
            host_v_dva, resident_k.data_ptr(), resident_v.data_ptr(),
            block_size, k_bytes, v_bytes, max_token, self.device,
        )
        self.assertEqual(result, 0)
        torch.npu.synchronize()

        self.assertEqual(last_req_ids.cpu().tolist(),
                         expected["last_req_ids"])
        self.assertEqual(slot_to_token.cpu().tolist(),
                         expected["slot_to_token"])
        self.assertEqual(lru_slots.cpu().tolist(), expected["lru_slots"])
        self.assertEqual(current_slots.cpu().tolist(),
                         expected["current_slots"])
        self.assertEqual(miss_count.cpu().tolist(), expected["miss_count"])
        self.assertEqual(miss_tokens.cpu().tolist(), expected["miss_tokens"])
        self.assertEqual(miss_slots.cpu().tolist(), expected["miss_slots"])

        resident_k_cpu = resident_k.cpu()
        resident_v_cpu = resident_v.cpu()
        for row in range(num_reqs):
            copied_slots = set()
            for index in range(expected["miss_count"][row]):
                token = expected["miss_tokens"][row][index]
                slot = expected["miss_slots"][row][index]
                copied_slots.add(slot)
                block = case["block_table"][row][token // block_size]
                physical_token = block * block_size + token % block_size
                self.assertTrue(torch.equal(
                    resident_k_cpu[row, slot], host_k[physical_token]
                ))
                self.assertTrue(torch.equal(
                    resident_v_cpu[row, slot], host_v[physical_token]
                ))
            for slot in range(capacity):
                if slot not in copied_slots:
                    self.assertTrue(torch.count_nonzero(
                        resident_k_cpu[row, slot]
                    ).item() == 0)
                    self.assertTrue(torch.count_nonzero(
                        resident_v_cpu[row, slot]
                    ).item() == 0)

    def test_runtime_shape_and_multibatch_matrix(self):
        matrix = [
            (1, 1, 1, 16),
            (2, 31, 47, 128),
            # More valid misses than resident slots: verifies that only
            # assigned misses are published and the output tail stays -1.
            (2, 33, 7, 128),
            (3, 33, 97, 8192),
            (8, 513, 777, 32768),
        ]
        for shape in matrix:
            with self.subTest(shape=shape):
                self.run_case(*shape)


if __name__ == "__main__":
    unittest.main()
