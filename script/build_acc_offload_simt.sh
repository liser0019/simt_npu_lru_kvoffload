#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
project_dir=$(cd "${script_dir}/.." && pwd)
operator_dir="${project_dir}/src/acc_offload/csrc/operators"
output_dir=${1:-"${project_dir}/output/acc_offload/production"}
simt_npu_arch=${ASCEND_SIMT_NPU_ARCH:-dav-3510}
aiv_npu_arch=${ASCEND_AIV_NPU_ARCH:-dav-c310}

if ! command -v bisheng >/dev/null 2>&1; then
    echo "ERROR: bisheng not found; source the CANN environment first." >&2
    exit 1
fi
mkdir -p "${output_dir}"

build_device_library() {
    local output_library=$1
    local kernel_kind=$2
    shift 2
    local sources=("$@")
    local flags=()
    local link_flags=()
    local objects=()

    if [[ "${kernel_kind}" == "simt" ]]; then
        flags=(--npu-arch="${simt_npu_arch}" --enable-simt)
        link_flags=("${flags[@]}")
    else
        flags=(--cce-aicore-arch="${aiv_npu_arch}")
    fi

    local source object
    for source in "${sources[@]}"; do
        object="${output_dir}/${output_library%.so}_${source%.cpp}.o"
        echo "Building ${source} -> ${output_library} (${flags[*]})"
        bisheng -x asc "${operator_dir}/${source}" -fPIC -c -O2 \
            "${flags[@]}" -o "${object}"
        objects+=("${object}")
    done
    bisheng -shared -fPIC -O2 "${objects[@]}" "${link_flags[@]}" \
        -o "${output_dir}/${output_library}"
    rm -f "${objects[@]}"
}

aiv_kernel_library="libmf_hybm_accoffload_kernel_aiv.so"
simt_kernel_library="libmf_hybm_accoffload_kernel_simt.so"
rm -f "${output_dir}/${aiv_kernel_library}" \
      "${output_dir}/${simt_kernel_library}"

# dav-c310 library: registered-Host-safe DataCopyPad plus mixed SIMD/SIMT VFs.
build_device_library "${aiv_kernel_library}" aiv \
    acc_offload_sparse_copy.cpp \
    acc_offload_lru_compact_fused_v3.cpp \
    acc_offload_lru_resident_addrs_mixed_parallel.cpp

# dav-3510 library: generic fallbacks for shapes outside specialized kernels.
build_device_library "${simt_kernel_library}" simt \
    acc_offload_lru_compact_simt.cpp \
    acc_offload_lru_resident_addrs_simt.cpp

cat > "${output_dir}/acc_offload_kernel_impl.info" <<EOF
mode=production
aiv_npu_arch=${aiv_npu_arch}
simt_npu_arch=${simt_npu_arch}
kernel_library_aiv=${aiv_kernel_library}
kernel_library_simt=${simt_kernel_library}
sparse_copy=aiv_datacopypad
lru_resident_compact=mixed_ub_fused_v3
lru_compact_fallback=simt_generic
lru_compact_v3_shape=topk:2048,capacity:4096,max_token:4096
lru_compact_v3_threads=1024
lru_compact_v3_dynamic_ub_bytes=114720
lru_compact_v3_input_bytes=40960
lru_compact_v3_output_bytes=57360
lru_compact_v3_scan=warp_prefix
lru_compact_v3_output_path=simt_direct_gm
compute_lru_resident_addrs=mixed_simt_parallel
resident_addrs_fallback=simt_serial
resident_addrs_parallel_shape=num_reqs:1,max_topk:2048
resident_addrs_parallel_threads=1024
resident_addrs_parallel_dynamic_ub_bytes=288
resident_addrs_parallel_scan=warp_prefix
EOF

if ! command -v nm >/dev/null 2>&1; then
    echo "ERROR: nm not found; cannot verify device symbols." >&2
    exit 1
fi
has_exact_symbol() {
    local library=$1 expected=$2
    nm -D --defined-only "${library}" | awk -v expected="${expected}" '
        NF > 0 && $NF == expected { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

has_exact_symbol "${output_dir}/${aiv_kernel_library}" OffloadOpsSparseCopy
has_exact_symbol "${output_dir}/${aiv_kernel_library}" OffloadOpsLruCompactFusedV3
has_exact_symbol "${output_dir}/${aiv_kernel_library}" OffloadOpsComputeLruResidentAddrsMixedParallel
! has_exact_symbol "${output_dir}/${aiv_kernel_library}" OffloadOpsLruCompact
! has_exact_symbol "${output_dir}/${aiv_kernel_library}" OffloadOpsComputeLruResidentAddrs
has_exact_symbol "${output_dir}/${simt_kernel_library}" OffloadOpsLruCompact
has_exact_symbol "${output_dir}/${simt_kernel_library}" OffloadOpsComputeLruResidentAddrs
! has_exact_symbol "${output_dir}/${simt_kernel_library}" OffloadOpsSparseCopy

python3 "${project_dir}/test/python/acc_offload/test_simt_operators.py"
echo "A5 production kernel build and CPU/source tests passed."
