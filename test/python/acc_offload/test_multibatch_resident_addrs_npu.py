#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""Focused CPU-oracle and A5 tests for ResidentAddrs Multi-Batch V2.

The pure-Python oracle tests run everywhere.  Device tests are opt-in so this
file remains importable on developer hosts without torch_npu/CANN:

    MF_RUN_A5_RESIDENT_MULTIBATCH=1 python3 \
        test/python/acc_offload/test_multibatch_resident_addrs_npu.py -v

Optional latency sampling (10 warmups, 20 repeats by default):

    MF_RUN_A5_RESIDENT_MULTIBATCH=1 \
    MF_RUN_A5_RESIDENT_MULTIBATCH_PERF=1 \
        python3 test/python/acc_offload/test_multibatch_resident_addrs_npu.py -v

The output ABI is globally packed ``[all K][all V]``.  It is not grouped as
``req0 K/V, req1 K/V``.  The matrix intentionally uses request-specific block
tables and destination rows so missing request strides fail deterministically.
"""

import os
import statistics
import time
import unittest


RUN_A5 = os.getenv("MF_RUN_A5_RESIDENT_MULTIBATCH") == "1"
RUN_PERF = os.getenv("MF_RUN_A5_RESIDENT_MULTIBATCH_PERF") == "1"

GVAS_SENTINEL = -101
ADDR_SENTINEL = -202
SIZE_SENTINEL = -303

if RUN_A5:
    import torch
    import torch_npu  # noqa: F401
    from memfabric_hybrid import offload


def resident_addrs_oracle(case):
    """Return the exact request-major, stable ``[all K][all V]`` output."""
    k_gvas, v_gvas = [], []
    k_addrs, v_addrs = [], []
    k_sizes, v_sizes = [], []
    block_bytes_k = case["block_size"] * case["token_bytes_k"]
    block_bytes_v = case["block_size"] * case["token_bytes_v"]

    for req in range(case["num_reqs"]):
        count = max(0, min(case["miss_count"][req], case["topk"]))
        for index in range(count):
            token = case["miss_tokens"][req][index]
            slot = case["miss_slots"][req][index]
            if token < 0 or slot < 0 or slot >= case["resident_capacity"]:
                continue
            block_id = token // case["block_size"]
            if block_id < 0 or block_id >= case["max_num_blocks"]:
                continue
            block_index = case["block_table"][req][block_id]
            if block_index < 0:
                continue

            offset = token % case["block_size"]
            linear_slot = req * case["resident_capacity"] + slot
            k_gvas.append(
                case["gvas_k_base"] + block_index * block_bytes_k
                + offset * case["token_bytes_k"]
            )
            v_gvas.append(
                case["gvas_v_base"] + block_index * block_bytes_v
                + offset * case["token_bytes_v"]
            )
            k_addrs.append(
                case["addr_k_base"] + linear_slot * case["token_bytes_k"]
            )
            v_addrs.append(
                case["addr_v_base"] + linear_slot * case["token_bytes_v"]
            )
            k_sizes.append(case["token_bytes_k"])
            v_sizes.append(case["token_bytes_v"])

    return {
        "num_tokens": 2 * len(k_gvas),
        "gvas": k_gvas + v_gvas,
        "addrs": k_addrs + v_addrs,
        "sizes": k_sizes + v_sizes,
    }


def make_case(num_reqs, topk, requested_counts=None, all_valid=False):
    """Build deterministic heterogeneous rows with visible request strides."""
    block_size = 4
    max_num_blocks = 64
    resident_capacity = 128
    count_pattern = [
        min(topk, 33), topk, 0, 1, min(topk, 31), min(topk, 32),
        min(topk, 512), min(topk, 1025), topk + 7, -3,
    ]
    miss_count = list(requested_counts) if requested_counts is not None else [
        count_pattern[req % len(count_pattern)] for req in range(num_reqs)
    ]
    if len(miss_count) != num_reqs:
        raise ValueError("requested_counts must have num_reqs entries")

    block_table = []
    miss_tokens = []
    miss_slots = []
    for req in range(num_reqs):
        # A large per-request delta makes a missing block-table row stride
        # obvious in descriptor values.
        table = [req * 1000 + block + 10 for block in range(max_num_blocks)]
        if not all_valid:
            table[7] = -1
        block_table.append(table)

        tokens, slots = [], []
        for index in range(topk):
            token = (req * 17 + index * 3) % (max_num_blocks * block_size - 1)
            slot = (req * 11 + index * 7) % resident_capacity
            if index % 19 == 0:
                token = 5  # duplicates are valid and must not be deduplicated
            if not all_valid:
                selector = index % 37
                if selector == 1:
                    token = -1
                elif selector == 2:
                    token = max_num_blocks * block_size  # invalid block id
                elif selector == 3:
                    slot = -1
                elif selector == 4:
                    slot = resident_capacity
                elif selector == 5:
                    token = 7 * block_size  # block_table[req][7] == -1
            tokens.append(token)
            slots.append(slot)
        miss_tokens.append(tokens)
        miss_slots.append(slots)

    return {
        "num_reqs": num_reqs,
        "topk": topk,
        "block_size": block_size,
        "max_num_blocks": max_num_blocks,
        "resident_capacity": resident_capacity,
        "token_bytes_k": 8,
        "token_bytes_v": 16,
        "gvas_k_base": 1_000_000,
        "gvas_v_base": 5_000_000,
        "addr_k_base": 9_000_000,
        "addr_v_base": 13_000_000,
        "miss_count": miss_count,
        "miss_tokens": miss_tokens,
        "miss_slots": miss_slots,
        "block_table": block_table,
    }


class TestResidentAddrsCpuOracle(unittest.TestCase):
    """Always-on tests prove packing, invalid filtering and request strides."""

    def test_hand_computed_two_request_example(self):
        case = {
            "num_reqs": 2, "topk": 3, "block_size": 4,
            "max_num_blocks": 3, "resident_capacity": 4,
            "token_bytes_k": 8, "token_bytes_v": 16,
            "gvas_k_base": 1000, "gvas_v_base": 5000,
            "addr_k_base": 9000, "addr_v_base": 13000,
            "miss_count": [3, 3],
            "miss_tokens": [[1, -1, 8], [4, 5, 6]],
            "miss_slots": [[2, 1, 0], [1, 4, 3]],
            "block_table": [[10, 11, -1], [20, 21, 22]],
        }
        self.assertEqual(resident_addrs_oracle(case), {
            "num_tokens": 6,
            "gvas": [1328, 1672, 1688, 5656, 6344, 6376],
            "addrs": [9016, 9040, 9056, 13032, 13080, 13112],
            "sizes": [8, 8, 8, 16, 16, 16],
        })

    def test_request_major_all_k_all_v_layout(self):
        case = make_case(3, 33, requested_counts=[33, 17, 9])
        actual = resident_addrs_oracle(case)
        self.assertEqual(actual["num_tokens"], len(actual["gvas"]))
        self.assertEqual(len(actual["gvas"]), len(actual["addrs"]))
        self.assertEqual(len(actual["gvas"]), len(actual["sizes"]))
        half = actual["num_tokens"] // 2
        self.assertTrue(all(size == case["token_bytes_k"]
                            for size in actual["sizes"][:half]))
        self.assertTrue(all(size == case["token_bytes_v"]
                            for size in actual["sizes"][half:]))

    def test_clamp_duplicate_invalid_and_stride(self):
        case = make_case(4, 32, requested_counts=[-3, 39, 32, 1])
        actual = resident_addrs_oracle(case)
        self.assertGreater(actual["num_tokens"], 0)
        # Row 0 is clamped to zero.  All K destinations emitted for later rows
        # must include at least one resident-capacity request stride.
        half = actual["num_tokens"] // 2
        self.assertTrue(all(
            address >= case["addr_k_base"]
            + case["resident_capacity"] * case["token_bytes_k"]
            for address in actual["addrs"][:half]
        ))


@unittest.skipUnless(RUN_A5, "set MF_RUN_A5_RESIDENT_MULTIBATCH=1 on A5")
class TestResidentAddrsMultiBatchA5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        extend_path = os.getenv("MEMFABRIC_HYBRID_EXTEND_LIB_PATH")
        if not extend_path:
            raise RuntimeError("source the target MemFabric set_env.sh first")
        manifest_path = os.path.join(extend_path, "acc_offload_kernel_impl.info")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            cls.manifest = dict(
                line.strip().split("=", 1) for line in manifest_file
                if "=" in line
            )
        if cls.manifest.get("compute_lru_resident_addrs") != \
                "mixed_simt_parallel":
            raise RuntimeError(f"wrong ResidentAddrs backend: {cls.manifest}")
        if cls.manifest.get("resident_addrs_parallel_version") != "2":
            raise RuntimeError(f"wrong ResidentAddrs version: {cls.manifest}")
        print(f"[RESIDENT_MULTIBATCH_MANIFEST] {cls.manifest}", flush=True)

    def npu_tensor(self, data, dtype):
        return torch.tensor(data, dtype=dtype, device=self.device)

    def run_case(self, case):
        output_capacity = 2 * case["num_reqs"] * case["topk"]
        miss_count = self.npu_tensor(case["miss_count"], torch.int32)
        miss_tokens = self.npu_tensor(case["miss_tokens"], torch.int32)
        miss_slots = self.npu_tensor(case["miss_slots"], torch.int32)
        block_table = self.npu_tensor(case["block_table"], torch.int32)
        gvas = torch.full(
            (output_capacity,), GVAS_SENTINEL, dtype=torch.int64,
            device=self.device,
        )
        addrs = torch.full(
            (output_capacity,), ADDR_SENTINEL, dtype=torch.int64,
            device=self.device,
        )
        sizes = torch.full(
            (output_capacity,), SIZE_SENTINEL, dtype=torch.int32,
            device=self.device,
        )
        num_tokens = torch.full((1,), -1, dtype=torch.int32, device=self.device)

        self.assertEqual(offload.compute_lru_resident_addrs(
            miss_count, miss_tokens, miss_slots, block_table, gvas, addrs,
            sizes, num_tokens, case["block_size"], case["token_bytes_k"],
            case["token_bytes_v"], case["gvas_k_base"], case["gvas_v_base"],
            case["addr_k_base"], case["addr_v_base"],
            case["resident_capacity"], case["num_reqs"], case["topk"],
            case["max_num_blocks"], self.device,
        ), 0)
        torch.npu.synchronize()

        expected = resident_addrs_oracle(case)
        actual_num_tokens = int(num_tokens.cpu().item())
        actual_gvas = gvas.cpu().tolist()
        actual_addrs = addrs.cpu().tolist()
        actual_sizes = sizes.cpu().tolist()
        self.assertEqual(actual_num_tokens, expected["num_tokens"])
        self.assertEqual(actual_gvas[:actual_num_tokens], expected["gvas"])
        self.assertEqual(actual_addrs[:actual_num_tokens], expected["addrs"])
        self.assertEqual(actual_sizes[:actual_num_tokens], expected["sizes"])
        self.assertTrue(all(value == GVAS_SENTINEL
                            for value in actual_gvas[actual_num_tokens:]))
        self.assertTrue(all(value == ADDR_SENTINEL
                            for value in actual_addrs[actual_num_tokens:]))
        self.assertTrue(all(value == SIZE_SENTINEL
                            for value in actual_sizes[actual_num_tokens:]))

    def test_multibatch_correctness_matrix(self):
        matrix = [
            (1, 1), (2, 31), (3, 32), (4, 33), (8, 511), (9, 512),
            (16, 513), (3, 1023), (4, 1024), (2, 1025), (16, 2048),
        ]
        for num_reqs, topk in matrix:
            with self.subTest(num_reqs=num_reqs, topk=topk):
                self.run_case(make_case(num_reqs, topk))

    def test_batch1_regression_counts(self):
        for count in (0, 512, 1024, 1536, 2048):
            with self.subTest(count=count):
                self.run_case(make_case(1, 2048, requested_counts=[count]))

    @staticmethod
    def latency_summary(samples_ms):
        ordered = sorted(samples_ms)
        size = len(ordered)
        at = lambda q: ordered[min(size - 1, int((size - 1) * q))]
        return {
            "mean": statistics.mean(ordered), "min": ordered[0],
            "p50": at(0.50), "p95": at(0.95), "p99": at(0.99),
            "max": ordered[-1],
        }

    @unittest.skipUnless(
        RUN_PERF, "set MF_RUN_A5_RESIDENT_MULTIBATCH_PERF=1 for latency",
    )
    def test_multibatch_latency(self):
        warmup = int(os.getenv("MF_A5_RESIDENT_MULTIBATCH_WARMUP", "10"))
        repeats = int(os.getenv("MF_A5_RESIDENT_MULTIBATCH_REPEATS", "20"))
        topk = 2048
        for num_reqs in (1, 2, 4, 8, 16):
            for workload, count in (
                ("all_miss", topk), ("half_miss", topk // 2),
                ("zero_miss", 0),
            ):
                case = make_case(
                    num_reqs, topk, requested_counts=[count] * num_reqs,
                    all_valid=True,
                )
                output_capacity = 2 * num_reqs * topk
                miss_count = self.npu_tensor(case["miss_count"], torch.int32)
                miss_tokens = self.npu_tensor(case["miss_tokens"], torch.int32)
                miss_slots = self.npu_tensor(case["miss_slots"], torch.int32)
                block_table = self.npu_tensor(case["block_table"], torch.int32)
                gvas = torch.empty(output_capacity, dtype=torch.int64,
                                   device=self.device)
                addrs = torch.empty_like(gvas)
                sizes = torch.empty(output_capacity, dtype=torch.int32,
                                    device=self.device)
                num_tokens = torch.zeros(1, dtype=torch.int32,
                                         device=self.device)

                def submit():
                    result = offload.compute_lru_resident_addrs(
                        miss_count, miss_tokens, miss_slots, block_table,
                        gvas, addrs, sizes, num_tokens, case["block_size"],
                        case["token_bytes_k"], case["token_bytes_v"],
                        case["gvas_k_base"], case["gvas_v_base"],
                        case["addr_k_base"], case["addr_v_base"],
                        case["resident_capacity"], num_reqs, topk,
                        case["max_num_blocks"], self.device,
                    )
                    self.assertEqual(result, 0)

                for _ in range(warmup):
                    submit()
                torch.npu.synchronize()
                samples = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    submit()
                    torch.npu.synchronize()
                    samples.append((time.perf_counter() - started) * 1000.0)

                summary = self.latency_summary(samples)
                print(
                    "RESIDENT_MULTIBATCH_PERF "
                    f"num_reqs={num_reqs} topk={topk} workload={workload} "
                    f"mean_ms={summary['mean']:.6f} "
                    f"us_per_req={summary['mean'] * 1000.0 / num_reqs:.3f} "
                    f"min_ms={summary['min']:.6f} p50_ms={summary['p50']:.6f} "
                    f"p95_ms={summary['p95']:.6f} p99_ms={summary['p99']:.6f} "
                    f"max_ms={summary['max']:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
