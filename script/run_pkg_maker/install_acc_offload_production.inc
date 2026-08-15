# This file is sourced by install.sh inside try_install_extend().
# It builds the A5 production configuration:
#   - descriptor-free SparseKvLoadRuntime (Plan + Transfer)
#   - legacy three-operator path retained only for A/B and rollback
#   - AIV/DataCopyPad SparseCopy
#   - Compact V3 direct-GM
#   - ResidentAddrs Parallel V2 with a dedicated batch=1 fast path
#   - generic SIMT fallbacks for unsupported shapes

function build_acc_offload_device_library()
{
    local output_library=$1
    local kernel_kind=$2
    shift 2
    local kernel_sources="$*"
    local compile_params="${cce_param}"
    local link_params=""
    local kernel_objects=""
    local kernel_source
    local kernel_object

    if [ "${kernel_kind}" == "simt" ]; then
        compile_params="${simt_compile_params}"
        link_params="${simt_compile_params}"
    fi

    rm -f "${output_library}"
    for kernel_source in ${kernel_sources}; do
        kernel_object="${output_library%.so}_${kernel_source%.cpp}.o"
        print "INFO" "building ${kernel_source} for ${output_library} with ${compile_params}"
        if ! bisheng -x asc "${kernel_source}" -fPIC -c -g -o "${kernel_object}" \
            ${compile_params}; then
            rm -f ${kernel_objects} "${kernel_object}" "${output_library}"
            return 1
        fi
        kernel_objects="${kernel_objects} ${kernel_object}"
    done

    if ! bisheng -shared -fPIC -g -o "${output_library}" ${kernel_objects} \
        ${link_params}; then
        rm -f ${kernel_objects} "${output_library}"
        return 1
    fi
    rm -f ${kernel_objects}
    return 0
}

function has_exact_device_symbol()
{
    local library=$1
    local symbol=$2
    nm -D --defined-only "${library}" 2>/dev/null |
        awk -v expected="${symbol}" '
            NF > 0 && $NF == expected { found = 1 }
            END { exit(found ? 0 : 1) }
        '
}

function require_device_symbol()
{
    has_exact_device_symbol "$1" "$2"
}

function reject_device_symbol()
{
    ! has_exact_device_symbol "$1" "$2"
}

bisheng_path=$(which bisheng 2>/dev/null)
if [ -z "${bisheng_path}" ]; then
    print "ERROR" "bisheng Not Found; cannot build accoffload kernels."
    exit 1
fi

aiv_npu_arch=${ASCEND_AIV_NPU_ARCH:-}
if [ -z "${aiv_npu_arch}" ]; then
    case "${SOC_VERSION:-}" in
        ascend950*|Ascend950*|ASCEND950*) aiv_npu_arch="dav-c310" ;;
    esac
fi
if [ -z "${aiv_npu_arch}" ] && [ "${ascend_version}" = "A5" ]; then
    aiv_npu_arch="dav-c310"
fi
if [ -z "${aiv_npu_arch}" ]; then
    print "ERROR" "Set ASCEND_AIV_NPU_ARCH=dav-c310 or SOC_VERSION for A5."
    exit 1
fi
simt_npu_arch=${ASCEND_SIMT_NPU_ARCH:-dav-3510}
cce_param="--cce-aicore-arch=${aiv_npu_arch}"
simt_compile_params="--npu-arch=${simt_npu_arch} --enable-simt"

cd ${script_dir}/../copy_extend
if bisheng -x asc hybm_copy_kernel.cpp -fPIC -shared -g \
    -o libmf_hybm_copy_extend.so ${cce_param}; then
    cp ./*.so ${install_dir}/${pkg_arch}-${os1}/lib64
else
    print "WARNING" "install extend lib failed; CANN may be too old"
fi

if [ ! -d "${script_dir}/../accoffload_operators" ]; then
    print "ERROR" "accoffload_operators not found in run package."
    exit 1
fi
cd ${script_dir}/../accoffload_operators

aiv_kernel_library="libmf_hybm_accoffload_kernel_aiv.so"
simt_kernel_library="libmf_hybm_accoffload_kernel_simt.so"
kernel_libraries="${aiv_kernel_library} ${simt_kernel_library}"
rm -f ${kernel_libraries} acc_offload_operators_launch.o \
    libmf_hybm_accoffload.so

if ! build_acc_offload_device_library "${aiv_kernel_library}" aiv \
    acc_offload_sparse_copy.cpp acc_offload_lru_compact_fused_v3.cpp \
    acc_offload_lru_resident_addrs_mixed_parallel.cpp \
    acc_offload_sparse_kv_plan_runtime.cpp \
    acc_offload_sparse_kv_transfer_runtime.cpp; then
    print "ERROR" "build production AIV/mixed kernel library failed."
    exit 1
fi
if ! build_acc_offload_device_library "${simt_kernel_library}" simt \
    acc_offload_lru_compact_simt.cpp \
    acc_offload_lru_resident_addrs_simt.cpp; then
    print "ERROR" "build generic SIMT fallback library failed."
    rm -f "${aiv_kernel_library}"
    exit 1
fi

if ! require_device_symbol "${aiv_kernel_library}" OffloadOpsSparseCopy ||
   ! require_device_symbol "${aiv_kernel_library}" OffloadOpsLruCompactFusedV3 ||
   ! require_device_symbol "${aiv_kernel_library}" OffloadOpsComputeLruResidentAddrsMixedParallel ||
   ! require_device_symbol "${aiv_kernel_library}" OffloadOpsSparseKvPlanRuntime ||
   ! require_device_symbol "${aiv_kernel_library}" OffloadOpsSparseKvPlanFsaRuntime ||
   ! require_device_symbol "${aiv_kernel_library}" OffloadOpsSparseKvTransferRuntime ||
   ! reject_device_symbol "${aiv_kernel_library}" OffloadOpsLruCompact ||
   ! reject_device_symbol "${aiv_kernel_library}" OffloadOpsComputeLruResidentAddrs ||
   ! require_device_symbol "${simt_kernel_library}" OffloadOpsLruCompact ||
   ! require_device_symbol "${simt_kernel_library}" OffloadOpsComputeLruResidentAddrs ||
   ! reject_device_symbol "${simt_kernel_library}" OffloadOpsSparseCopy; then
    print "ERROR" "production device symbols are not isolated as expected."
    rm -f ${kernel_libraries}
    exit 1
fi

python_bin=python3
torch_dir=$(${python_bin} -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
torch_npu_dir=$(${python_bin} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))" 2>/dev/null)
if [ -z "${torch_dir}" ] || [ -z "${torch_npu_dir}" ]; then
    print "ERROR" "torch/torch_npu not found; cannot build Host launcher."
    rm -f ${kernel_libraries}
    exit 1
fi
ascend_home=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
abi_flag="-D_GLIBCXX_USE_CXX11_ABI=0"
abi_val=$(${python_bin} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null)
if [ "${abi_val}" = "1" ]; then abi_flag="-D_GLIBCXX_USE_CXX11_ABI=1"; fi

if ! g++ -c -fPIC -std=c++17 -O3 -fstack-protector-strong \
    -Wno-unused-parameter -Wno-unused-function -Wunused-value -Wcast-align \
    -Wcast-qual -Wwrite-strings -Wsign-compare -Wextra \
    -fvisibility-inlines-hidden -ftrapv ${abi_flag} \
    -DMF_ACC_OFFLOAD_PRODUCTION=1 \
    -isystem ${ascend_home}/include \
    -isystem ${ascend_home}/include/experiment/runtime/runtime/ \
    -isystem ${torch_dir}/include \
    -isystem ${torch_dir}/include/torch/csrc/api/include \
    -isystem ${torch_dir}/include/torch/csrc/utils \
    -isystem ${torch_dir}/include/c10/util \
    -isystem ${torch_dir}/include/c10/core \
    -isystem ${torch_dir}/include/ATen \
    -isystem ${torch_dir}/include/ATen/detail \
    -isystem ${torch_npu_dir}/include \
    -isystem ${torch_npu_dir}/include/torch_npu/csrc/aten \
    -isystem ${torch_npu_dir}/include/torch_npu/csrc/core/npu \
    acc_offload_operators_launch.cpp -o acc_offload_operators_launch.o; then
    print "ERROR" "compile production Host launcher failed."
    rm -f ${kernel_libraries} acc_offload_operators_launch.o
    exit 1
fi

if ! g++ -shared -fPIC -o libmf_hybm_accoffload.so \
    acc_offload_operators_launch.o \
    -Wl,--no-as-needed -L. -lmf_hybm_accoffload_kernel_aiv \
    -lmf_hybm_accoffload_kernel_simt -Wl,--as-needed \
    -L${torch_dir}/lib -ltorch -lc10 -ltorch_python \
    -L${torch_npu_dir}/lib -ltorch_npu \
    -L${ascend_home}/lib64 -lopapi \
    -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now \
    -Wl,-rpath,\$ORIGIN -Wl,-rpath,${torch_dir}/lib \
    -Wl,-rpath,${torch_npu_dir}/lib -Wl,-rpath,${ascend_home}/lib64; then
    print "ERROR" "link production Host library failed."
    rm -f ${kernel_libraries} acc_offload_operators_launch.o \
        libmf_hybm_accoffload.so
    exit 1
fi

accoffload_lib_dir=${install_dir}/${pkg_arch}-${os1}/lib64
if ! \cp libmf_hybm_accoffload.so ${kernel_libraries} "${accoffload_lib_dir}"; then
    print "ERROR" "install production accoffload libraries failed."
    rm -f ${kernel_libraries} acc_offload_operators_launch.o \
        libmf_hybm_accoffload.so
    exit 1
fi
impl_info=${accoffload_lib_dir}/acc_offload_kernel_impl.info
cat > "${impl_info}" <<EOF
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
lru_compact_v3_scan=warp_prefix
lru_compact_v3_output_path=simt_direct_gm
compute_lru_resident_addrs=mixed_simt_parallel
resident_addrs_fallback=simt_serial
resident_addrs_parallel_version=2
resident_addrs_parallel_shape=num_reqs:dynamic,max_topk:2048
resident_addrs_parallel_threads=1024
resident_addrs_parallel_dynamic_ub_bytes=288
resident_addrs_parallel_scan=warp_prefix
resident_addrs_parallel_batch1_fast_path=true
resident_addrs_parallel_row_schedule=serial_rows_parallel_misses
sparse_kv_load_runtime=plan_plus_transfer
sparse_kv_plan=mixed_simt_runtime_hash
sparse_kv_plan_shape=num_reqs:runtime,topk:runtime,capacity:runtime,max_token:validation_only
sparse_kv_plan_threads=1024
sparse_kv_plan_dynamic_ub_bytes=288
sparse_kv_plan_workspace=O(num_reqs*(topk+capacity))
sparse_kv_plan_max_token_workspace=false
sparse_kv_plan_hash=gm_open_addressed_cas
sparse_kv_plan_row_schedule=max_32_blocks_block_stride
sparse_kv_fsa_plan_runtime=mixed_simt_hash_stable_rows
sparse_kv_fsa_plan_output=npu_int16_external_plan_v1
sparse_kv_fsa_plan_mapping=device_deterministic_two_pass
sparse_kv_fsa_plan_transfer_runtime=false
sparse_kv_fsa_plan_sync=one_stream_no_host_sync
sparse_kv_transfer=aiv_datacopypad
sparse_kv_transfer_blocks=32
sparse_kv_transfer_descriptor_materialization=false
sparse_kv_load_runtime_sync=one_stream_no_intermediate_sync
EOF

for artifact in libmf_hybm_accoffload.so ${kernel_libraries}; do
    rm -f "${artifact}"
done
rm -f acc_offload_operators_launch.o
