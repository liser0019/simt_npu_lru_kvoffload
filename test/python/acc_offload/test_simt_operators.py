#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""CPU semantics and source-contract tests for the public production kernels.

The real kernels require CANN/Bisheng and an A5 device.  This lightweight file
is intentionally runnable on an ordinary build host: it checks the important
algorithmic contracts and verifies that packaging exposes only the selected
production implementations plus the two documented generic fallbacks.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
OPERATORS = REPO / "src/acc_offload/csrc/operators"


def stable_exclusive_ranks(flags):
    total = 0
    ranks = []
    for flag in flags:
        ranks.append(total)
        total += int(bool(flag))
    return ranks, total


def resident_addrs_reference(tokens, slots, block_table, *, block_size,
                             capacity, max_blocks, token_bytes_k,
                             token_bytes_v, gvas_k, gvas_v, addr_k, addr_v):
    """Stable filter/address oracle matching ComputeLruResidentAddrs."""
    valid = []
    for token, slot in zip(tokens, slots):
        if token < 0 or slot < 0 or slot >= capacity:
            continue
        block_id = token // block_size
        if block_id < 0 or block_id >= max_blocks:
            continue
        block_index = block_table[block_id]
        if block_index < 0:
            continue
        offset = token % block_size
        valid.append((slot, block_index, offset))

    k_gvas, v_gvas, k_addrs, v_addrs = [], [], [], []
    for slot, block_index, offset in valid:
        k_gvas.append(gvas_k + block_index * block_size * token_bytes_k +
                      offset * token_bytes_k)
        v_gvas.append(gvas_v + block_index * block_size * token_bytes_v +
                      offset * token_bytes_v)
        k_addrs.append(addr_k + slot * token_bytes_k)
        v_addrs.append(addr_v + slot * token_bytes_v)
    return {
        "task_count": 2 * len(valid),
        "gvas": k_gvas + v_gvas,
        "addrs": k_addrs + v_addrs,
        "sizes": [token_bytes_k] * len(valid) +
                 [token_bytes_v] * len(valid),
    }


def test_warp_prefix_contract():
    flags = [int(index % 7 not in (1, 5)) for index in range(2048)]
    ranks, total = stable_exclusive_ranks(flags)
    packed = [index for index, flag in enumerate(flags) if flag]
    reconstructed = [None] * total
    for index, flag in enumerate(flags):
        if flag:
            reconstructed[ranks[index]] = index
    assert reconstructed == packed


def test_resident_addrs_stable_filter_and_kv_layout():
    # Includes a duplicate token and all invalid categories. Duplicate valid
    # entries are preserved; this operator filters but never deduplicates.
    result = resident_addrs_reference(
        [5, 5, -1, 20, 6, 9], [2, 3, 0, 1, -1, 1],
        [4, 7, -1, 3], block_size=4, capacity=4, max_blocks=4,
        token_bytes_k=8, token_bytes_v=16,
        gvas_k=1000, gvas_v=5000, addr_k=10000, addr_v=20000,
    )
    assert result["task_count"] == 4
    assert result["gvas"] == [1232, 1232, 5464, 5464]
    assert result["addrs"] == [10016, 10024, 20032, 20048]
    assert result["sizes"] == [8, 8, 16, 16]


def test_production_operator_source_contract():
    compact = (OPERATORS / "acc_offload_lru_compact_fused_v3.cpp").read_text()
    compact_header = (OPERATORS / "acc_offload_lru_compact_fused_v3.h").read_text()
    resident = (OPERATORS / "acc_offload_lru_resident_addrs_mixed_parallel.cpp").read_text()
    sparse_copy = (OPERATORS / "acc_offload_sparse_copy.cpp").read_text()

    assert "OffloadOpsLruCompactFusedV3" in compact
    assert "OffloadOpsLruCompactFusedV3" in compact_header
    assert "LAUNCH_BOUND(LRU_FUSED_V3_THREADS)" in compact
    assert "asc_vf_call<LruCompactFusedV3Vf>" in compact
    assert "asc_atomic_min" in compact and "asc_atomic_max" in compact
    assert "FUSED_V3_DYNAMIC_UB_BYTES == 114720U" in compact
    assert "V_MTE3" not in compact

    assert "OffloadOpsComputeLruResidentAddrsMixedParallel" in resident
    assert "RESIDENT_ADDRS_THREADS = 1024" in resident
    assert "WarpExclusiveScan1024" in resident
    assert "RESIDENT_ADDRS_DYNAMIC_UB_BYTES == 288U" in resident

    assert "OffloadOpsSparseCopy" in sparse_copy
    assert "constexpr uint32_t blockDim = 32" in sparse_copy
    assert "OffloadSparseCopyOps<<<blockDim" in sparse_copy


def test_public_tree_contains_only_baseline_and_current_operator_sources():
    present = {path.name for path in OPERATORS.iterdir() if path.is_file()}
    required = {
        "README_SIMT.md",
        "acc_offload_lru_compact_fused_v3.cpp",
        "acc_offload_lru_compact_fused_v3.h",
        "acc_offload_lru_compact_simt.cpp",
        "acc_offload_lru_resident_addrs_mixed_parallel.cpp",
        "acc_offload_lru_resident_addrs_mixed_parallel.h",
        "acc_offload_lru_resident_addrs_simt.cpp",
        "acc_offload_operators.h",
        "acc_offload_sparse_copy.cpp",
        "acc_offload_sparse_copy.h",
    }
    # release_kv_miss already contains the generic AIV Compact/ResidentAddrs
    # sources used by its original build.  Keep those baseline files to avoid
    # an unrelated framework deletion; the production run-package source list
    # below does not compile them into the optimized A5 libraries.
    assert required <= present
    forbidden_experiments = {
        "acc_offload_lru_compact_mixed_ub.cpp",
        "acc_offload_lru_compact_mixed_ub.h",
        "acc_offload_lru_compact_fused_ub.cpp",
        "acc_offload_lru_compact_fused_ub.h",
        "acc_offload_lru_compact_fused_ub_v2.cpp",
        "acc_offload_lru_compact_fused_ub_v2.h",
        "acc_offload_sparse_copy_simt.cpp",
    }
    assert present.isdisjoint(forbidden_experiments)


def test_host_dispatch_and_build_are_production_only():
    host = (REPO / "src/acc_offload/csrc/launch/acc_offload_operators_launch.cpp").read_text()
    build = (REPO / "script/build_acc_offload_simt.sh").read_text()
    install = (REPO / "script/run_pkg_maker/install_acc_offload_production.inc").read_text()
    package = (REPO / "script/run_pkg_maker/make_run.sh").read_text()

    for source in (host, build, install):
        assert "OffloadOpsLruCompactFusedV3" in source
        assert "OffloadOpsComputeLruResidentAddrsMixedParallel" in source
    assert "MF_LRU_COMPACT_PLAN_IMPL" not in host
    assert "MF_LRU_RESIDENT_ADDRS_IMPL" not in host
    assert "MF_ACC_OFFLOAD_KERNEL_IMPL" not in build
    assert "mode=production" in build and "mode=production" in install
    assert "install_acc_offload_production.inc" in package
    assert "csrc/operators/*" not in package
    assert "acc_offload_lru_compact_fused_v3.cpp" in package
    assert "acc_offload_lru_resident_addrs_mixed_parallel.cpp" in package


def run_all():
    tests = [
        test_warp_prefix_contract,
        test_resident_addrs_stable_filter_and_kv_layout,
        test_production_operator_source_contract,
        test_public_tree_contains_only_baseline_and_current_operator_sources,
        test_host_dispatch_and_build_are_production_only,
    ]
    for test in tests:
        test()
    print(f"PUBLIC_PRODUCTION_SOURCE_TESTS_PASS cases={len(tests)}")


if __name__ == "__main__":
    run_all()
