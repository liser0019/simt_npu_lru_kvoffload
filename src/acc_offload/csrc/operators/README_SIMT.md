# A5 Sparse KV Offload production kernels

This public tree intentionally contains one current production implementation
per operator, plus generic SIMT fallbacks for unsupported shapes.

## Included implementations

| Operator | Production implementation | Generic fallback |
|---|---|---|
| SparseCopy | AIV `DataCopyPad` | none |
| LRU Compact | fused V3 direct-GM | SIMT generic |
| ResidentAddrs | 1024-lane mixed parallel | SIMT serial |

Production sources:

```text
acc_offload_sparse_copy.cpp/.h
acc_offload_lru_compact_fused_v3.cpp/.h
acc_offload_lru_resident_addrs_mixed_parallel.cpp/.h
```

Fallback sources:

```text
acc_offload_lru_compact_simt.cpp
acc_offload_lru_resident_addrs_simt.cpp
```

The fallback kernels are not historical benchmark variants. They preserve the
public API for shapes outside the specialized production contracts.

## Automatic dispatch

Users do not select experimental backends with environment variables.
`acc_offload_operators_launch.cpp` dispatches automatically:

```text
Compact:
  topk=2048, capacity=4096, max_token=4096 -> fused V3
  any other shape                            -> generic SIMT

ResidentAddrs:
  num_reqs=1, 0<topk<=2048 -> mixed parallel
  any other shape          -> serial SIMT
```

`MF_LRU_COMPACT_PLAN_IMPL`, `MF_LRU_RESIDENT_ADDRS_IMPL`, and the historical
`MF_ACC_OFFLOAD_KERNEL_IMPL` selector are no longer used by this cleaned tree.

The optional `MF_LRU_COMPACT_V3_PROFILE=1` switch remains available. It only
enables device stage timestamps; it does not select a different algorithm.

## Device libraries

The build remains split because the kernels target different A5 execution
frontends:

```text
dav-c310 AIV/mixed library:
  SparseCopy + Compact V3 + ResidentAddrs Parallel

dav-3510 --enable-simt library:
  generic Compact + generic ResidentAddrs
```

Build on an A5/CANN host:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export SOC_VERSION=ascend950pr_957b
export ASCEND_AIV_NPU_ARCH=dav-c310
export ASCEND_SIMT_NPU_ARCH=dav-3510
bash script/build_acc_offload_simt.sh
```

The generated `acc_offload_kernel_impl.info` must contain
`mode=production`. A run package uses the same source list through
`script/run_pkg_maker/install_acc_offload_production.inc`.

## Important contracts

- Compact V3 uses 1024 lanes, 114720 bytes dynamic UB, warp-prefix stable
  compaction, UB atomic first/last ownership, and direct GM publication.
- ResidentAddrs maps up to 2048 misses onto 1024 lanes, validates each item
  once, performs two stable warp-prefix scans, and emits
  `[all K descriptors][all V descriptors]`.
- SparseCopy consumes an even number of paired K/V tasks:
  `task_count = 2 * valid_miss_count`.
- Registered Host DVA copy remains on AIV/DataCopyPad; scalar SIMT Host copy is
  deliberately not included.

Historical V1/V2, mixed-plan, thread-sweep and MTE3 experiment sources were
removed from the public production tree. The pre-cleanup snapshot is not
required to build or use this repository.
