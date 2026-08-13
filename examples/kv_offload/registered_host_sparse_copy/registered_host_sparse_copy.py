#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""A5 single-rank registered Host DDR sparse-copy round-trip test.

Run this test before starting vLLM. A 507035/vector-core exception poisons the
current device process; stop and restart the container before another test.
"""

import argparse
import os

import torch
import torch_npu
import memfabric_hybrid as mf
from memfabric_hybrid import offload


ONE_GIB = 1 << 30
COPY_SIZES = (1, 7, 8, 31, 32, 33, 128, 257, 1024)
UNALIGNED_CASES = ((1, 8), (1, 32), (3, 128))
REPEATS = 3


def _load_kernel_mode() -> str:
    extend_lib_path = os.environ.get("MEMFABRIC_HYBRID_EXTEND_LIB_PATH")
    if not extend_lib_path:
        raise RuntimeError(
            "MEMFABRIC_HYBRID_EXTEND_LIB_PATH is unset; source set_env.sh for "
            "the selected installation"
        )
    manifest_path = os.path.join(extend_lib_path, "acc_offload_kernel_impl.info")
    with open(manifest_path, encoding="utf-8") as manifest:
        values = dict(
            line.strip().split("=", 1)
            for line in manifest
            if "=" in line
        )
    mode = values.get("mode", "")
    print(f"[KERNEL_IMPLEMENTATION] path={manifest_path} values={values}", flush=True)
    if mode != "production":
        raise RuntimeError(f"unsupported or missing installed kernel mode: {mode!r}")
    return mode


def _copy(
    sources: list[int],
    destinations: list[int],
    copy_lengths: list[int],
    device: torch.device,
) -> None:
    if not (len(sources) == len(destinations) == len(copy_lengths)):
        raise ValueError("SparseCopy descriptor arrays must have equal lengths")
    source_tensor = torch.tensor(sources, dtype=torch.int64, device=device)
    destination_tensor = torch.tensor(destinations, dtype=torch.int64, device=device)
    length_tensor = torch.tensor(copy_lengths, dtype=torch.int32, device=device)
    count = torch.tensor([len(copy_lengths)], dtype=torch.int32, device=device)
    result = offload.sparse_copy(
        source_tensor,
        destination_tensor,
        length_tensor,
        count,
        device,
    )
    if result not in (None, 0):
        raise RuntimeError(f"offload.sparse_copy returned {result}")
    torch.npu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    _load_kernel_mode()

    torch.npu.set_device(args.device)
    mf.set_log_level(3)
    config = offload.OffloadConfig()
    config.device_id = args.device
    config.reserve_size = ONE_GIB
    config.alloc_size = ONE_GIB
    config.world_size = 1
    config.rank_id = 0
    config.scene = offload.Scene.SHARED
    config.register_host_memory = True
    if offload.initialize(config) != 0:
        raise RuntimeError("offload.initialize failed")

    try:
        hosts = [
            offload.empty([copy_size], dtype=torch.uint8).zero_()
            for copy_size in COPY_SIZES
        ]
        host_device_addresses = [
            int(offload.get_device_address(host)) for host in hosts
        ]
        if any(address == 0 for address in host_device_addresses):
            raise RuntimeError("offload.get_device_address returned a null Host DVA")
        device = torch.device("npu", args.device)
        for host, host_device_address in zip(hosts, host_device_addresses):
            print(
                "[REGISTERED_HOST] "
                f"cpu=0x{host.data_ptr():x} device=0x{host_device_address:x} "
                f"size={host.numel()} device_id={args.device}",
                flush=True,
            )

        for repeat in range(REPEATS):
            sources = [
                (torch.arange(size, dtype=torch.int32, device=device) + repeat * 17 + index)
                .remainder_(251)
                .to(torch.uint8)
                for index, size in enumerate(COPY_SIZES)
            ]
            destinations = [torch.zeros_like(source) for source in sources]
            for host in hosts:
                host.zero_()

            _copy(
                [source.data_ptr() for source in sources],
                host_device_addresses,
                list(COPY_SIZES),
                device,
            )
            for size, host, source in zip(COPY_SIZES, hosts, sources):
                torch.npu.synchronize()
                if not torch.equal(host, source.cpu()):
                    raise AssertionError(
                        f"NPU to registered Host DDR failed: repeat={repeat}, size={size}"
                    )

            _copy(
                host_device_addresses,
                [destination.data_ptr() for destination in destinations],
                list(COPY_SIZES),
                device,
            )
            for size, destination, host in zip(COPY_SIZES, destinations, hosts):
                torch.npu.synchronize()
                if not torch.equal(destination.cpu(), host):
                    raise AssertionError(
                        f"registered Host DDR to NPU failed: repeat={repeat}, size={size}"
                    )
            print(
                f"[PASS] registered Host DDR bidirectional batch repeat={repeat}",
                flush=True,
            )

        guard_bytes = 5
        unaligned_hosts = [
            offload.empty([offset + size + guard_bytes], dtype=torch.uint8).zero_()
            for offset, size in UNALIGNED_CASES
        ]
        unaligned_host_dvas = [
            int(offload.get_device_address(host)) for host in unaligned_hosts
        ]
        for repeat in range(REPEATS):
            sources = [
                (
                    torch.arange(offset + size + guard_bytes, dtype=torch.int32,
                                 device=device)
                    + repeat * 29
                    + index
                ).remainder_(251).to(torch.uint8)
                for index, (offset, size) in enumerate(UNALIGNED_CASES)
            ]
            destinations = [torch.zeros_like(source) for source in sources]
            for host in unaligned_hosts:
                host.zero_()

            _copy(
                [source.data_ptr() + offset
                 for source, (offset, _) in zip(sources, UNALIGNED_CASES)],
                [address + offset
                 for address, (offset, _) in zip(unaligned_host_dvas, UNALIGNED_CASES)],
                [size for _, size in UNALIGNED_CASES],
                device,
            )
            for (offset, size), host, source in zip(
                    UNALIGNED_CASES, unaligned_hosts, sources):
                torch.npu.synchronize()
                if not torch.equal(
                        host[offset:offset + size], source.cpu()[offset:offset + size]):
                    raise AssertionError(
                        "NPU to unaligned registered Host DDR failed: "
                        f"repeat={repeat}, offset={offset}, size={size}"
                    )
                torch.npu.synchronize()
                if not torch.equal(host[:offset], torch.zeros_like(host[:offset])):
                    raise AssertionError("NPU to Host copy overwrote the prefix guard")
                torch.npu.synchronize()
                if not torch.equal(
                        host[offset + size:], torch.zeros_like(host[offset + size:])):
                    raise AssertionError("NPU to Host copy overwrote the suffix guard")

            _copy(
                [address + offset
                 for address, (offset, _) in zip(unaligned_host_dvas, UNALIGNED_CASES)],
                [destination.data_ptr() + offset
                 for destination, (offset, _) in zip(destinations, UNALIGNED_CASES)],
                [size for _, size in UNALIGNED_CASES],
                device,
            )
            for (offset, size), destination, host in zip(
                    UNALIGNED_CASES, destinations, unaligned_hosts):
                torch.npu.synchronize()
                if not torch.equal(
                        destination.cpu()[offset:offset + size], host[offset:offset + size]):
                    raise AssertionError(
                        "unaligned registered Host DDR to NPU failed: "
                        f"repeat={repeat}, offset={offset}, size={size}"
                    )
                torch.npu.synchronize()
                actual = destination.cpu()
                if not torch.equal(actual[:offset], torch.zeros_like(actual[:offset])):
                    raise AssertionError("Host to NPU copy overwrote the prefix guard")
                torch.npu.synchronize()
                if not torch.equal(
                        actual[offset + size:], torch.zeros_like(actual[offset + size:])):
                    raise AssertionError("Host to NPU copy overwrote the suffix guard")
            print(
                f"[PASS] unaligned registered Host DDR batch repeat={repeat}",
                flush=True,
            )

        print("REGISTERED_HOST_SPARSE_COPY_PASS", flush=True)
    finally:
        offload.uninitialize()


if __name__ == "__main__":
    main()
