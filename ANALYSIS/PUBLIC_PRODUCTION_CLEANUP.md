# Public production cleanup

## Scope

The original working tree carried every optimization experiment used to reach
the current A5 implementation. Public users only need the final production
path and safe generic fallbacks, so this tree was reduced to:

```text
LruCompact V3
    -> ComputeLruResidentAddrs Parallel V1
    -> AIV/DataCopyPad SparseCopy
```

## Runtime data flow

```text
Python public API
  -> acc_offload_operators_launch.cpp
     -> production-shape gate
        -> optimized dav-c310 mixed kernel
        -> otherwise generic dav-3510 SIMT fallback
```

SparseCopy does not need a fallback because the included AIV/DataCopyPad path
is the registered-Host-safe production implementation.

## Removed implementation families

- AIV-only all-operator mode
- scalar SIMT SparseCopy
- mixed-UB Plan-only Compact
- fused Compact V1
- Compact V2 and V2-warp public entries
- V3 MTE3 publication experiment
- runtime selectors for the historical experiment matrix

## Why two fallback sources remain

Compact V3 is compiled for `topk=2048/capacity=4096/max_token=4096`.
ResidentAddrs Parallel V1 supports `num_reqs=1/topk<=2048`. Removing their
fallbacks would make the existing void public ABI unable to report unsupported
shapes safely. The fallbacks therefore are compatibility code, not selectable
historical versions.

## Validation status

CPU/source tests and shell/Python syntax can run on a development host. The
renamed V3-only translation unit and changed build/install source lists require
a fresh Bisheng compile and A5 correctness run before publishing a release
binary. Do not infer device validation from source-only checks.
