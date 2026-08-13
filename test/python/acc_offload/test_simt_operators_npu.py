#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

"""A5 真机上的 acc-offload 算子正确性与性能测试。

这不是普通的 CPU 单元测试，而是安装 MemFabric run package 后，在真实
Ascend950PR/A5 上调用 device kernel 的测试入口。默认不会导入 torch_npu，
只有显式设置 ``MF_RUN_A5_SIMT_TESTS=1`` 才会访问 NPU，因而开发机也能安全地
执行本文件并看到全部用例被 skip。

建议按下面的顺序阅读本文件：

1. ``setUpClass``：选择 NPU、初始化 registered-host 内存池，并读取安装包
   manifest，确认实际运行的是 AIV、SIMT 还是 mixed backend。
2. ``new_production_lru_state``：构造 production shape
   ``num_reqs/topk/capacity/max_token = 1/2048/4096/4096`` 的 LRU 状态。
3. ``run_production_lru``：公共 LRU Compact Python ABI 的集中调用点。
4. ``new_sparse_kv_pipeline_case``：构造三算子流水线所需的真实 NPU/registered
   Host 缓冲区、地址表以及可验证的 K/V 数据 pattern。
5. ``submit_sparse_kv_pipeline``：按同一条当前 NPU stream 异步提交
   ``LruCompact -> ComputeLruResidentAddrs -> SparseCopy``，中间不读 CPU 标量。
6. ``assert_sparse_kv_pipeline_result``：在最终 synchronize 后同时核验 descriptor
   布局和最终 K/V 字节，防止“地址看起来正确但 K/V 或 slot 搬错”。
7. ``test_lru_*``：LRU 小尺寸语义、production correctness、不同 hit ratio 性能。
8. ``test_sparse_kv_*``：三算子端到端正确性和两种计时口径。
9. ``test_resident_addresses`` / ``test_sparse_copy_*``：后两个算子的独立边界测试。

三算子的关键 device-side 数据流是：

``LruCompact``
    输出 ``miss_count / miss_tokens / miss_slots``
        ↓（仍在 NPU 上）
``ComputeLruResidentAddrs``
    输出 ``[全部K地址][全部V地址]``、size 和偶数 ``task_count=2*miss_count``
        ↓（仍在 NPU 上）
``SparseCopy``
    根据地址表完成 Host↔NPU 或 NPU↔NPU 的真实字节搬运
        ↓
``torch.npu.synchronize()`` 后才进行 CPU 断言。

常用运行方式：

.. code-block:: bash

    MF_RUN_A5_SIMT_TESTS=1 python3 \
        test/python/acc_offload/test_simt_operators_npu.py -v

    MF_RUN_A5_SIMT_TESTS=1 \
    MF_RUN_A5_SPARSE_KV_PIPELINE=1 \
    python3 test/python/acc_offload/test_simt_operators_npu.py -v

注意：性能用例中的 state 构造、registered-host 分配和 CPU correctness 检查均不在
计时区间内。stage 口径每个算子后同步一次；pipeline 口径只在三个算子都提交后
同步一次，二者回答的问题不同，不能混为同一个性能数字。
"""

import hashlib
import json
import os
import random
import statistics
import time
import unittest


# 总开关。不开启时不导入 torch/torch_npu，避免没有 NPU/CANN 的开发机 import 失败。
RUN_A5_TESTS = os.getenv("MF_RUN_A5_SIMT_TESTS") == "1"


def load_installed_kernel_manifest():
    """读取当前已安装包的 kernel manifest，防止测错 device library。

    ``set_env.sh`` 会设置 ``MEMFABRIC_HYBRID_EXTEND_LIB_PATH``。该目录下的
    ``acc_offload_kernel_impl.info`` 记录 SparseCopy、LRU、ResidentAddrs 分别使用
    AIV/SIMT 的哪种实现。测试启动时打印完整内容，服务器日志因此能够证明实际
    测到的 backend，而不是仅凭环境变量推断。
    """
    extend_lib_path = os.getenv("MEMFABRIC_HYBRID_EXTEND_LIB_PATH")
    if not extend_lib_path:
        raise RuntimeError(
            "MEMFABRIC_HYBRID_EXTEND_LIB_PATH is unset; source the selected "
            "MemFabric installation's set_env.sh before running NPU tests"
        )
    manifest_path = os.path.join(extend_lib_path, "acc_offload_kernel_impl.info")
    values = {}
    with open(manifest_path, encoding="utf-8") as manifest:
        for line in manifest:
            key, separator, value = line.strip().partition("=")
            if separator:
                values[key] = value
    if values.get("mode") != "production":
        raise RuntimeError(f"invalid or missing kernel mode in {manifest_path}: {values}")
    return manifest_path, values

# 延迟导入：只有真机测试打开时才要求 torch_npu 和安装后的 memfabric_hybrid。
if RUN_A5_TESTS:
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    from memfabric_hybrid import offload


@unittest.skipUnless(RUN_A5_TESTS, "set MF_RUN_A5_SIMT_TESTS=1 on an A5 host")
class TestA5SimtOperators(unittest.TestCase):
    """A5 上 acc-offload 三个算子及其组合流水线的测试集合。"""

    @classmethod
    def setUpClass(cls):
        """整个测试类只初始化一次 NPU 和 MemFabric registered-host 内存池。"""
        cls.manifest_path, cls.kernel_manifest = load_installed_kernel_manifest()
        cls.kernel_mode = cls.kernel_manifest["mode"]
        print(
            f"[KERNEL_IMPLEMENTATION] path={cls.manifest_path} "
            f"values={cls.kernel_manifest}",
            flush=True,
        )
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        config = offload.OffloadConfig()
        config.device_id = 0
        # correctness pattern 远小于 64 MiB；这里留出充足 registered-host 空间，
        # 使 Host 内存可以通过 get_device_address() 获得 NPU 可访问的 DVA。
        config.reserve_size = 64 << 20
        config.alloc_size = 64 << 20
        config.world_size = 1
        config.rank_id = 0
        config.scene = offload.Scene.LOCAL
        config.register_host_memory = True
        if offload.initialize(config) != 0:
            raise RuntimeError("offload.initialize failed")

    @classmethod
    def tearDownClass(cls):
        """释放 MemFabric 全局资源，避免影响同进程后续测试。"""
        offload.uninitialize()

    def npu_tensor(self, data, dtype):
        """小型测试数据的快捷构造器，统一放到当前 A5 device。"""
        return torch.tensor(data, dtype=dtype, device=self.device)

    def new_production_lru_state(self):
        """构造互不共享 storage 的 production-shape fresh/new-request 状态。

        ``last_req_ids=-1`` 而 ``req_ids=20260806``，因此第一次 Compact 必然走
        new-request/reset path。TopK 是 0..2047，cache 初始为空，所以 2048 个位置
        都是 miss，并依次分配 slot 0..2047。

        返回 dict 是为了让不同测试能够 clone/覆盖单个 tensor，同时复用统一 ABI。
        ``token_mark/token_pos/epochs`` 虽在 V2/V3 中主要是内部 workspace/兼容参数，
        仍按公共 ABI 完整提供。
        """
        num_reqs, topk, capacity, max_token = 1, 2048, 4096, 4096
        return {
            "num_reqs": num_reqs,
            "topk": topk,
            "capacity": capacity,
            "max_token": max_token,
            # req_id 不同表示 fresh/new request；相同则表示 steady-state continuation。
            "req_ids": self.npu_tensor([20260806], torch.int64),
            "last_req_ids": self.npu_tensor([-1], torch.int64),
            "topk_indices": torch.arange(topk, dtype=torch.int32).view(1, -1).to(self.device),
            # stable_prefix=max_token 表示本次不因 prefix 门槛淘汰任何 resident token。
            "stable_prefix": self.npu_tensor([max_token], torch.int32),
            "slot_to_token": torch.full(
                (num_reqs, capacity), -1, dtype=torch.int32, device=self.device
            ),
            "lru_slots": torch.arange(capacity, dtype=torch.int32).view(1, -1).to(self.device),
            "current_slots": torch.full(
                (num_reqs, topk), -1, dtype=torch.int32, device=self.device
            ),
            "miss_count": torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
            "miss_tokens": torch.full(
                (num_reqs, topk), -1, dtype=torch.int32, device=self.device
            ),
            "miss_slots": torch.full(
                (num_reqs, topk), -1, dtype=torch.int32, device=self.device
            ),
            "token_mark": torch.zeros((8, max_token), dtype=torch.int32, device=self.device),
            "token_pos": torch.full(
                (8, max_token), -1, dtype=torch.int32, device=self.device
            ),
            "epochs": torch.zeros(8, dtype=torch.int32, device=self.device),
        }

    def run_production_lru(self, state):
        """调用公共 LRU Compact ABI；函数只负责提交，不负责 synchronize。

        不在这里同步非常重要：单算子测试可以在调用后自行同步，而 pipeline 测试
        可以继续在同一 stream 提交 ResidentAddrs 和 SparseCopy，实现真实异步链。
        """
        return offload.lru_resident_compact(
            state["req_ids"],
            state["last_req_ids"],
            state["topk_indices"],
            state["stable_prefix"],
            state["slot_to_token"],
            state["lru_slots"],
            state["current_slots"],
            state["miss_count"],
            state["miss_tokens"],
            state["miss_slots"],
            state["token_mark"],
            state["token_pos"],
            state["epochs"],
            state["num_reqs"],
            state["topk"],
            state["capacity"],
            state["max_token"],
            self.device,
        )

    @staticmethod
    def latency_summary(latencies_ms):
        """把毫秒样本汇总为 mean/min/p50/p95/p99/max。

        百分位使用 nearest-rank 风格的离散索引，避免为了测试日志引入 numpy。
        所有 A/B backend 都经过同一函数，比较口径保持一致。
        """
        ordered = sorted(latencies_ms)

        def percentile(percent):
            index = round((len(ordered) - 1) * percent / 100.0)
            return ordered[index]

        return {
            "mean": statistics.fmean(ordered),
            "min": ordered[0],
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
            "max": ordered[-1],
        }

    @staticmethod
    def format_latency_summary(summary):
        """格式化原有 LRU ``submit+sync`` 性能日志字段。"""
        return " ".join(
            f"{name}_submit_sync_ms={summary[name]:.3f}"
            for name in ("mean", "min", "p50", "p95", "p99", "max")
        )

    def new_production_partial_hit_state(self):
        """构造恰好 1024 hit + 1024 miss 的 production normal-path 状态。"""
        return self.new_production_hit_count_state(1024)

    def new_production_hit_count_state(self, hit_count):
        """构造 TopK 前 ``hit_count`` 个 token 已 resident 的 steady-state 状态。

        将 ``last_req_ids`` 改为当前请求可避免 reset fast path；slot 0..hit_count-1
        预装相同编号 token，其余 slot 为空。因此 hit/miss 数量能够精确控制。
        """
        state = self.new_production_lru_state()
        topk = state["topk"]
        capacity = state["capacity"]
        if not 0 <= hit_count <= topk:
            raise ValueError("hit_count must be within the production Top-K")
        state["last_req_ids"].fill_(20260806)
        resident = torch.cat(
            (
                torch.arange(hit_count, dtype=torch.int32),
                torch.full((capacity - hit_count,), -1, dtype=torch.int32),
            )
        ).view(1, -1).to(self.device)
        state["slot_to_token"].copy_(resident)
        return state

    @staticmethod
    def format_prefixed_latency_summary(prefix, summary):
        """为 pipeline 的 lru/addr/copy/pipeline 四组统计增加字段前缀。"""
        return " ".join(
            f"{prefix}_{name}_ms={summary[name]:.3f}"
            for name in ("mean", "min", "p50", "p95", "p99", "max")
        )

    @staticmethod
    def sparse_kv_pattern(token_count, token_bytes, salt):
        """生成可由 ``token id + byte offset`` 唯一推导的 CPU K/V pattern。

        K 和 V 使用不同 salt，所以即使长度恰好相同，K/V 被互换也会被断言发现。
        模 251 避免 uint8 周期与常见 256 对齐模式过度重合。
        """
        tokens = torch.arange(token_count, dtype=torch.int32).view(-1, 1)
        offsets = torch.arange(token_bytes, dtype=torch.int32).view(1, -1)
        return ((tokens * 37 + offsets * 13 + salt) % 251).to(torch.uint8)

    def new_sparse_kv_pipeline_case(self, hit_count=None, direction="host_to_npu",
                                    topk_values=None, token_bytes_k=None,
                                    token_bytes_v=None):
        """构造一次真实三算子调用所需的所有 tensor 和地址空间。

        参数说明：

        * ``hit_count=None``：fresh/new request，预期 2048 miss；整数则表示已 resident
          的 TopK 前缀长度，可精确构造 25/50/75/100% hit。
        * ``direction``：``host_to_npu`` 模拟 reload，``npu_to_host`` 模拟 offload。
        * ``topk_values``：可注入 duplicate/invalid token，但长度仍必须为 2048，确保
          mixed V3 不因 shape guard fallback 到别的 backend。
        * K/V token bytes 默认 32/48，仅用于轻量 correctness；性能测试可通过环境
          变量设置真实线上值。

        返回的 ``case`` 同时保有实际 buffer 对象和其数值地址。必须保留对象引用，
        否则 Python GC 后 address descriptor 会变成悬空地址。
        """
        if direction not in {"host_to_npu", "npu_to_host"}:
            raise ValueError(f"unsupported SparseCopy direction: {direction}")
        if hit_count is None:
            state = self.new_production_lru_state()
            initial_hits = 0
        else:
            state = self.new_production_hit_count_state(hit_count)
            initial_hits = hit_count

        if topk_values is not None:
            if len(topk_values) != state["topk"]:
                raise ValueError("custom Top-K must preserve the production shape")
            state["topk_indices"].copy_(self.npu_tensor([topk_values], torch.int32))

        # logical token 先通过 block_table 映射到 physical block，再加 block 内偏移。
        # identity block_table 让 expected address 容易手算，同时仍走真实地址算子。
        block_size = 16
        if token_bytes_k is None:
            token_bytes_k = int(os.getenv("MF_A5_SPARSE_KV_TOKEN_BYTES_K", "32"))
        if token_bytes_v is None:
            token_bytes_v = int(os.getenv("MF_A5_SPARSE_KV_TOKEN_BYTES_V", "48"))
        if token_bytes_k <= 0 or token_bytes_v <= 0:
            raise ValueError("K/V token byte sizes must be positive")
        max_num_blocks = state["max_token"] // block_size
        block_table = torch.arange(max_num_blocks, dtype=torch.int32).view(1, -1).to(
            self.device
        )
        # 每个有效 miss 产生一个 K task 和一个 V task，最坏情况是 2 * topk。
        # 这些 tensor 全在 NPU 上：后续 SparseCopy 直接消费，不经过 Python 重建。
        task_capacity = 2 * state["topk"]
        gvas_buffer = torch.zeros(task_capacity, dtype=torch.int64, device=self.device)
        addr_buffer = torch.zeros(task_capacity, dtype=torch.int64, device=self.device)
        size_buffer = torch.zeros(task_capacity, dtype=torch.int32, device=self.device)
        task_count = torch.zeros(1, dtype=torch.int32, device=self.device)

        source_k_cpu = self.sparse_kv_pattern(state["max_token"], token_bytes_k, 17)
        source_v_cpu = self.sparse_kv_pattern(state["max_token"], token_bytes_v, 109)
        # 目标 cache 先填充醒目的 sentinel。这样不仅能验证 miss 写对，还能验证
        # hit/all-hit 路径没有错误覆盖本不该 copy 的 slot。
        sentinel = 0xD3
        initial_dst_k = torch.full(
            (state["capacity"], token_bytes_k), sentinel, dtype=torch.uint8
        )
        initial_dst_v = torch.full(
            (state["capacity"], token_bytes_v), sentinel, dtype=torch.uint8
        )
        if initial_hits:
            # steady-state 的 resident slot 必须预先装入正确 K/V，否则 all-hit 即使
            # 不发生 copy，最终内容检查也会失败。这里只做测试状态准备，不计时。
            initial_dst_k[:initial_hits].copy_(source_k_cpu[:initial_hits])
            initial_dst_v[:initial_hits].copy_(source_v_cpu[:initial_hits])

        if direction == "host_to_npu":
            # Reload：源数据位于 registered Host。get_device_address() 返回 Host buffer
            # 对 NPU 可见的 DVA；目的 resident cache 是普通 NPU tensor。
            source_k = offload.empty([source_k_cpu.numel()], dtype=torch.uint8)
            source_v = offload.empty([source_v_cpu.numel()], dtype=torch.uint8)
            source_k.copy_(source_k_cpu.reshape(-1))
            source_v.copy_(source_v_cpu.reshape(-1))
            destination_k = initial_dst_k.reshape(-1).to(self.device)
            destination_v = initial_dst_v.reshape(-1).to(self.device)
            source_k_base = int(offload.get_device_address(source_k))
            source_v_base = int(offload.get_device_address(source_v))
            destination_k_base = destination_k.data_ptr()
            destination_v_base = destination_v.data_ptr()
        else:
            # Offload：源 KV 位于 NPU；目的使用 registered Host，并同样通过 DVA
            # 交给 ResidentAddrs/SparseCopy，绝不能把普通 CPU virtual address 传入。
            source_k = source_k_cpu.reshape(-1).to(self.device)
            source_v = source_v_cpu.reshape(-1).to(self.device)
            destination_k = offload.empty([initial_dst_k.numel()], dtype=torch.uint8)
            destination_v = offload.empty([initial_dst_v.numel()], dtype=torch.uint8)
            destination_k.copy_(initial_dst_k.reshape(-1))
            destination_v.copy_(initial_dst_v.reshape(-1))
            source_k_base = source_k.data_ptr()
            source_v_base = source_v.data_ptr()
            destination_k_base = int(offload.get_device_address(destination_k))
            destination_v_base = int(offload.get_device_address(destination_v))

        # 性能循环每轮都必须恢复完全相同的 LRU 初态，否则上一轮 miss 会让下一轮
        # 逐渐变成 all-hit，测到的是 workload 漂移而不是确定的 hit ratio。
        seed_names = (
            "req_ids", "last_req_ids", "topk_indices", "stable_prefix",
            "slot_to_token", "lru_slots", "current_slots", "miss_count",
            "miss_tokens", "miss_slots", "token_mark", "token_pos", "epochs",
        )
        return {
            "state": state,
            "state_seed": {name: state[name].clone() for name in seed_names},
            "direction": direction,
            "block_size": block_size,
            "token_bytes_k": token_bytes_k,
            "token_bytes_v": token_bytes_v,
            "max_num_blocks": max_num_blocks,
            "block_table": block_table,
            "gvas_buffer": gvas_buffer,
            "addr_buffer": addr_buffer,
            "size_buffer": size_buffer,
            "task_count": task_count,
            "source_k": source_k,
            "source_v": source_v,
            "source_k_cpu": source_k_cpu,
            "source_v_cpu": source_v_cpu,
            "destination_k": destination_k,
            "destination_v": destination_v,
            "initial_dst_k": initial_dst_k,
            "initial_dst_v": initial_dst_v,
            "source_k_base": source_k_base,
            "source_v_base": source_v_base,
            "destination_k_base": destination_k_base,
            "destination_v_base": destination_v_base,
        }

    def new_sparse_kv_shuffled_pipeline_case(self, direction):
        """构造 fixed-seed、非 identity resident/LRU 的 50% hit production case。

        TopK token、resident slot 和完整 LRU order 分别洗牌。测试因此不再依赖
        ``token==slot``、resident prefix 或 identity LRU 这些过于规则的巧合。
        固定 seed 让 A5 失败可稳定复现，并保证两个 copy 方向使用同一种业务状态。
        """
        case = self.new_sparse_kv_pipeline_case(0, direction)
        state = case["state"]
        rng = random.Random(20260812)

        topk_values = list(range(state["topk"]))
        rng.shuffle(topk_values)
        resident_tokens = topk_values[:1024]
        resident_slots = rng.sample(range(state["capacity"]), len(resident_tokens))
        resident = [-1] * state["capacity"]
        for token, slot in zip(resident_tokens, resident_slots):
            resident[slot] = token
        lru_order = list(range(state["capacity"]))
        rng.shuffle(lru_order)

        state["topk_indices"].copy_(self.npu_tensor([topk_values], torch.int32))
        state["slot_to_token"].copy_(self.npu_tensor([resident], torch.int32))
        state["lru_slots"].copy_(self.npu_tensor([lru_order], torch.int32))

        # 已 resident 的离散 slot 必须预装对应 token 的 K/V；其余仍保持 sentinel。
        initial_k = case["initial_dst_k"]
        initial_v = case["initial_dst_v"]
        for token, slot in zip(resident_tokens, resident_slots):
            initial_k[slot].copy_(case["source_k_cpu"][token])
            initial_v[slot].copy_(case["source_v_cpu"][token])
        case["destination_k"].copy_(initial_k.reshape(-1))
        case["destination_v"].copy_(initial_v.reshape(-1))

        for name in case["state_seed"]:
            case["state_seed"][name] = state[name].clone()
        return case

    @staticmethod
    def reset_sparse_kv_pipeline_case(case):
        """在计时区间外恢复 LRU 和 descriptor 输出，不重新分配大 buffer。"""
        state = case["state"]
        for name, seed in case["state_seed"].items():
            state[name].copy_(seed)
        case["gvas_buffer"].zero_()
        case["addr_buffer"].zero_()
        case["size_buffer"].zero_()
        case["task_count"].zero_()

    def submit_sparse_kv_lru(self, case):
        """只提交 LRU Compact；返回值 0 只表示 Host submit 成功。"""
        self.assertEqual(self.run_production_lru(case["state"]), 0)

    def submit_sparse_kv_addrs(self, case):
        """提交 ResidentAddrs，直接读取 LRU 写在 device 上的 miss 元数据。

        输出约定：前 ``N`` 项为 K descriptors，后 ``N`` 项为 V descriptors；
        ``task_count`` 由 device kernel 写成 ``2*N``。
        """
        state = case["state"]
        self.assertEqual(
            offload.compute_lru_resident_addrs(
                state["miss_count"], state["miss_tokens"], state["miss_slots"],
                case["block_table"], case["gvas_buffer"], case["addr_buffer"],
                case["size_buffer"], case["task_count"], case["block_size"],
                case["token_bytes_k"], case["token_bytes_v"],
                case["source_k_base"], case["source_v_base"],
                case["destination_k_base"], case["destination_v_base"],
                state["capacity"], state["num_reqs"], state["topk"],
                case["max_num_blocks"], self.device,
            ),
            0,
        )

    def submit_sparse_kv_copy(self, case):
        """提交 SparseCopy，直接消费 ResidentAddrs 的四个 device 输出。"""
        self.assertEqual(
            offload.sparse_copy(
                case["gvas_buffer"], case["addr_buffer"], case["size_buffer"],
                case["task_count"], self.device,
            ),
            0,
        )

    def submit_sparse_kv_pipeline(self, case):
        """在当前 NPU stream 上连续异步提交三算子，不插入 Host 同步。

        同一 stream 保证 kernel 顺序：Addr 必须等 LRU 写完 miss tensors，Copy 必须等
        Addr 写完 descriptors/task_count。这里若加入 CPU tensor 读取或显式同步，就会
        把本应纯 device-side 的依赖退化成 Host round trip。
        """
        self.submit_sparse_kv_lru(case)
        self.submit_sparse_kv_addrs(case)
        self.submit_sparse_kv_copy(case)

    @staticmethod
    def pipeline_destination_cpu(case, kind):
        """最终同步后，把 K/V resident cache 统一整理为 ``[slot, bytes]`` CPU 视图。"""
        destination = case[f"destination_{kind}"]
        token_bytes = case[f"token_bytes_{kind}"]
        if destination.device.type == "npu":
            destination = destination.cpu()
        return destination.reshape(case["state"]["capacity"], token_bytes)

    def assert_sparse_kv_pipeline_result(self, case):
        """从 descriptor 和最终字节两个层面验证三算子链。

        第一层验证：

        * task_count 必须严格等于 ``2 * miss_count``；
        * 地址顺序必须是 ``[all K][all V]``；
        * 每个 src/dst address 必须对应 LRU 给出的 token/slot；
        * K/V size 分别正确。

        第二层验证实际 copy 后的 resident cache。对每个 valid TopK，使用 LRU 的
        ``current_slots[position]`` 找到目标 slot，再与该 token 的 CPU K/V pattern
        比较。这样可以同时发现 token、slot、K/V 任一映射错误。
        """
        state = case["state"]
        miss_count = int(state["miss_count"].cpu().item())
        task_count = int(case["task_count"].cpu().item())
        self.assertEqual(task_count, 2 * miss_count)

        miss_tokens = state["miss_tokens"][0, :miss_count].cpu().tolist()
        miss_slots = state["miss_slots"][0, :miss_count].cpu().tolist()
        # ComputeLruResidentAddrs 的 ABI 是先放所有 K，再放所有 V，而不是 K0,V0 交错。
        expected_src = [
            case["source_k_base"] + token * case["token_bytes_k"]
            for token in miss_tokens
        ] + [
            case["source_v_base"] + token * case["token_bytes_v"]
            for token in miss_tokens
        ]
        expected_dst = [
            case["destination_k_base"] + slot * case["token_bytes_k"]
            for slot in miss_slots
        ] + [
            case["destination_v_base"] + slot * case["token_bytes_v"]
            for slot in miss_slots
        ]
        expected_sizes = (
            [case["token_bytes_k"]] * miss_count
            + [case["token_bytes_v"]] * miss_count
        )
        self.assertEqual(case["gvas_buffer"][:task_count].cpu().tolist(), expected_src)
        self.assertEqual(case["addr_buffer"][:task_count].cpu().tolist(), expected_dst)
        self.assertEqual(case["size_buffer"][:task_count].cpu().tolist(), expected_sizes)

        topk_values = state["topk_indices"][0].cpu().tolist()
        current_slots = state["current_slots"][0].cpu().tolist()
        destination_k = self.pipeline_destination_cpu(case, "k")
        destination_v = self.pipeline_destination_cpu(case, "v")
        # 不只检查 miss：hit slot 也必须保留 prime 时的数据，invalid token 则必须
        # 保持 current_slot=-1，不能产生越界 descriptor。
        for position, token in enumerate(topk_values):
            slot = current_slots[position]
            if not 0 <= token < state["max_token"]:
                self.assertEqual(slot, -1)
                continue
            self.assertGreaterEqual(slot, 0)
            self.assertTrue(torch.equal(
                destination_k[slot], case["source_k_cpu"][token]
            ))
            self.assertTrue(torch.equal(
                destination_v[slot], case["source_v_cpu"][token]
            ))

        if miss_count == 0:
            # all-hit 时 task_count=0，SparseCopy 应成为真正的 zero-copy no-op。
            self.assertTrue(torch.equal(destination_k, case["initial_dst_k"]))
            self.assertTrue(torch.equal(destination_v, case["initial_dst_v"]))

    @staticmethod
    def snapshot_external_lru_state(state):
        """同步后复制 Compact 的全部业务可见输出，排除仅内部使用的 workspace。"""
        names = (
            "last_req_ids",
            "slot_to_token",
            "lru_slots",
            "current_slots",
            "miss_count",
            "miss_tokens",
            "miss_slots",
        )
        return {name: state[name].cpu().tolist() for name in names}

    @staticmethod
    def external_lru_digest(snapshot):
        """把外部状态标准化为 SHA256，便于跨进程/backend 对照日志。"""
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ======================================================================
    # 第一组：LRU Compact 独立正确性与性能
    # ======================================================================

    def test_lru_compact(self):
        """小尺寸手算用例：同时覆盖 resident hit、miss 分配和稳定 LRU 重排。

        resident 为 ``slot->token=[1,3,5,-1]``，TopK 为 ``[3,6,1]``。
        token 3/1 命中，token 6 占用 slot 2；最终逐项断言所有外部状态。
        """
        num_reqs, topk, capacity, max_token = 1, 3, 4, 128
        req_ids = self.npu_tensor([10], torch.int64)
        last_req_ids = self.npu_tensor([10], torch.int64)
        topk_indices = self.npu_tensor([[3, 6, 1]], torch.int32)
        stable_prefix = self.npu_tensor([100], torch.int32)
        slot_to_token = self.npu_tensor([[1, 3, 5, -1]], torch.int32)
        lru_slots = self.npu_tensor([[0, 1, 2, 3]], torch.int32)
        current_slots = torch.full((1, topk), -1, dtype=torch.int32, device=self.device)
        miss_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        miss_tokens = torch.full((1, topk), -1, dtype=torch.int32, device=self.device)
        miss_slots = torch.full((1, topk), -1, dtype=torch.int32, device=self.device)
        token_mark = torch.zeros((8, max_token), dtype=torch.int32, device=self.device)
        token_pos = torch.zeros_like(token_mark)
        epochs = torch.zeros(8, dtype=torch.int32, device=self.device)

        self.assertEqual(offload.lru_resident_compact(
            req_ids, last_req_ids, topk_indices, stable_prefix, slot_to_token, lru_slots,
            current_slots, miss_count, miss_tokens, miss_slots, token_mark, token_pos, epochs,
            num_reqs, topk, capacity, max_token, self.device), 0)
        torch.npu.synchronize()

        self.assertEqual(current_slots.cpu().tolist(), [[1, 2, 0]])
        self.assertEqual(miss_count.cpu().tolist(), [1])
        self.assertEqual(miss_tokens.cpu().tolist()[0][0], 6)
        self.assertEqual(miss_slots.cpu().tolist()[0][0], 2)
        self.assertEqual(slot_to_token.cpu().tolist(), [[1, 3, 6, -1]])
        self.assertEqual(lru_slots.cpu().tolist(), [[3, 2, 0, 1]])

    def test_lru_compact_production_shape(self):
        """production fresh-state correctness、确定性及 submit+sync latency。

        每轮重新创建输入/输出/workspace，始终保持 2048 个 miss。若复用上一轮
        state，第 2..N 轮会变成 all-hit，从而污染 fresh fast-path 的性能数据。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")

        num_reqs, topk, capacity, max_token = 1, 2048, 4096, 4096
        expected_lru = torch.cat(
            (torch.arange(topk, capacity, dtype=torch.int32), torch.arange(topk, dtype=torch.int32))
        ).view(1, -1)
        expected_slot_to_token = torch.cat(
            (torch.arange(topk, dtype=torch.int32), torch.full((capacity - topk,), -1, dtype=torch.int32))
        ).view(1, -1)
        default_repeats = 100 if os.getenv("MF_RUN_A5_SIMT_STRESS") == "1" else 2
        repeats = int(os.getenv("MF_A5_SIMT_REPEATS", str(default_repeats)))
        if repeats <= 0:
            self.fail("MF_A5_SIMT_REPEATS must be a positive integer")
        synchronized_latencies_ms = []
        baseline = None

        for repeat in range(repeats):
            # state 构造在计时开始前；复用上一轮会把 fresh workload 变成 all-hit。
            state = self.new_production_lru_state()
            started = time.perf_counter()
            self.assertEqual(self.run_production_lru(state), 0)
            # device 执行是异步的；这里同步既是计时终点，也把异常归属到本次调用。
            torch.npu.synchronize()
            synchronized_latencies_ms.append((time.perf_counter() - started) * 1000.0)

            snapshot = self.snapshot_external_lru_state(state)
            self.assertEqual(snapshot["last_req_ids"], [20260806])
            self.assertEqual(snapshot["current_slots"], [list(range(topk))])
            self.assertEqual(snapshot["slot_to_token"], expected_slot_to_token.tolist())
            self.assertEqual(snapshot["lru_slots"], expected_lru.tolist())
            self.assertEqual(snapshot["miss_count"], [topk])
            self.assertEqual(snapshot["miss_tokens"], [list(range(topk))])
            self.assertEqual(snapshot["miss_slots"], [list(range(topk))])
            if baseline is None:
                baseline = snapshot
            else:
                self.assertEqual(snapshot, baseline)

        summary = self.latency_summary(synchronized_latencies_ms)
        print(
            "PRODUCTION_SIMT_LRU_FRESH_DETERMINISM_PASS "
            f"num_reqs={num_reqs} topk={topk} capacity={capacity} "
            f"max_token={max_token} repeats={repeats} "
            f"{self.format_latency_summary(summary)}",
            flush=True,
        )

    def test_lru_compact_production_shape_all_hit_performance(self):
        """先在计时外 prime cache，再测同一 request 的 steady all-hit 路径。

        每轮断言 ``miss_count=0``，避免实现错误悄悄改变实际 workload。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")
        default_repeats = 100 if os.getenv("MF_RUN_A5_SIMT_STRESS") == "1" else 2
        repeats = int(os.getenv("MF_A5_SIMT_PERF_REPEATS", str(default_repeats)))
        state = self.new_production_lru_state()
        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()  # Prime call is deliberately not timed.

        latencies = []
        for _ in range(repeats):
            started = time.perf_counter()
            self.assertEqual(self.run_production_lru(state), 0)
            torch.npu.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            self.assertEqual(state["miss_count"].cpu().tolist(), [0])
        print(
            "PRODUCTION_SIMT_LRU_ALL_HIT_PERF_PASS "
            f"repeats={repeats} "
            f"{self.format_latency_summary(self.latency_summary(latencies))}",
            flush=True,
        )

    def test_lru_compact_production_shape_partial_hit_performance(self):
        """测量可重复的 production 50% hit / 50% miss normal path。

        每轮在 timer 外重建 1024 resident + 1024 empty slot，避免上一轮 assignment
        改变下一轮的 hit ratio；最终状态还必须与第一轮完全一致。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")
        default_repeats = 100 if os.getenv("MF_RUN_A5_SIMT_STRESS") == "1" else 2
        repeats = int(os.getenv("MF_A5_SIMT_PERF_REPEATS", str(default_repeats)))
        expected_misses = 1024
        latencies = []
        baseline = None
        for _ in range(repeats):
            # State construction is outside the timed submit+sync interval.
            state = self.new_production_partial_hit_state()
            started = time.perf_counter()
            self.assertEqual(self.run_production_lru(state), 0)
            torch.npu.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            snapshot = self.snapshot_external_lru_state(state)
            self.assertEqual(snapshot["miss_count"], [expected_misses])
            if baseline is None:
                baseline = snapshot
            else:
                self.assertEqual(snapshot, baseline)
        print(
            "PRODUCTION_SIMT_LRU_PARTIAL_HIT_PERF_PASS "
            f"repeats={repeats} hits=1024 misses=1024 "
            f"{self.format_latency_summary(self.latency_summary(latencies))}",
            flush=True,
        )

    def test_lru_compact_production_shape_hit_ratio_performance(self):
        """可选 25%/75% hit 点，用于观察 normal-path latency 随 hit ratio 的变化。"""
        if os.getenv("MF_RUN_A5_SIMT_HIT_RATIO_SWEEP") != "1":
            self.skipTest("set MF_RUN_A5_SIMT_HIT_RATIO_SWEEP=1")
        repeats = int(os.getenv("MF_A5_SIMT_PERF_REPEATS", "100"))
        for hit_count in (512, 1536):
            expected_misses = 2048 - hit_count
            latencies = []
            baseline = None
            for _ in range(repeats):
                state = self.new_production_hit_count_state(hit_count)
                started = time.perf_counter()
                self.assertEqual(self.run_production_lru(state), 0)
                torch.npu.synchronize()
                latencies.append((time.perf_counter() - started) * 1000.0)
                snapshot = self.snapshot_external_lru_state(state)
                self.assertEqual(snapshot["miss_count"], [expected_misses])
                if baseline is None:
                    baseline = snapshot
                else:
                    self.assertEqual(snapshot, baseline)
            print(
                "PRODUCTION_SIMT_LRU_HIT_RATIO_PERF_PASS "
                f"repeats={repeats} hits={hit_count} misses={expected_misses} "
                f"{self.format_latency_summary(self.latency_summary(latencies))}",
                flush=True,
            )

    def test_lru_compact_v3_stage_profile(self):
        """读取 V3 可选的 device clock 时间戳，粗分正常路径阶段成本。

        第一次调用在 timer 外 prime 成 all-hit；第二次调用把 8 个 uint64 时间戳写入
        内部 token_mark workspace，测试输出 7 个相邻阶段的 cycle delta。
        """
        if os.getenv("MF_LRU_COMPACT_V3_PROFILE") != "1":
            self.skipTest("set MF_LRU_COMPACT_V3_PROFILE=1")
        state = self.new_production_lru_state()
        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()  # untimed prime creates the all-hit state
        state["token_mark"].zero_()
        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()
        stamps = state["token_mark"][0, :16].cpu().view(torch.int64).tolist()
        deltas = [stamps[index + 1] - stamps[index] for index in range(7)]
        self.assertTrue(all(delta >= 0 for delta in deltas))
        print(
            "PRODUCTION_SIMT_LRU_V3_STAGE_PROFILE_PASS "
            "stages=reset_prepare,topk_map,resident_owner,stable_compact,"
            "miss_or_all_hit,assign_rebuild,publication "
            f"cycle_deltas={','.join(str(delta) for delta in deltas)}",
            flush=True,
        )

    def test_lru_compact_production_shape_all_miss_then_hit(self):
        """同一 state 连续调用两次，验证第一次全 miss、第二次全 hit 的状态延续。

        该用例能发现 fused kernel 忘记发布 last_req_ids、LRU 或 resident state；
        单次 fresh correctness 无法发现这类跨调用错误。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")

        state = self.new_production_lru_state()
        topk = state["topk"]
        capacity = state["capacity"]
        expected_lru = [list(range(topk, capacity)) + list(range(topk))]
        expected_slot_to_token = [list(range(topk)) + [-1] * (capacity - topk)]

        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()
        first = self.snapshot_external_lru_state(state)
        self.assertEqual(first["last_req_ids"], [20260806])
        self.assertEqual(first["slot_to_token"], expected_slot_to_token)
        self.assertEqual(first["lru_slots"], expected_lru)
        self.assertEqual(first["current_slots"], [list(range(topk))])
        self.assertEqual(first["miss_count"], [topk])
        self.assertEqual(first["miss_tokens"], [list(range(topk))])
        self.assertEqual(first["miss_slots"], [list(range(topk))])

        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()
        second = self.snapshot_external_lru_state(state)
        self.assertEqual(second["last_req_ids"], [20260806])
        self.assertEqual(second["slot_to_token"], expected_slot_to_token)
        self.assertEqual(second["lru_slots"], expected_lru)
        self.assertEqual(second["current_slots"], [list(range(topk))])
        self.assertEqual(second["miss_count"], [0])
        self.assertEqual(second["miss_tokens"], [[-1] * topk])
        self.assertEqual(second["miss_slots"], [[-1] * topk])

    def test_lru_compact_production_shape_reset_fast_path_matrix(self):
        """production shape 下专测 reset fast path 的 duplicate/invalid 语义。

        ``4,4`` 是两个独立 miss；``-1`` 和 ``max_token`` 无效且不占 slot。
        valid position 稳定分配，最终 LRU 等于 identity LRU 的确定性旋转。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")
        state = self.new_production_lru_state()
        topk_values = [4, 4, -1, 4096, 8] + list(range(100, 2143))
        self.assertEqual(len(topk_values), state["topk"])
        state["topk_indices"].copy_(self.npu_tensor([topk_values], torch.int32))

        valid = [token for token in topk_values if 0 <= token < state["max_token"]]
        assign_count = len(valid)
        expected_current = []
        rank = 0
        for token in topk_values:
            if 0 <= token < state["max_token"]:
                expected_current.append(rank)
                rank += 1
            else:
                expected_current.append(-1)
        expected_slot_to_token = valid + [-1] * (state["capacity"] - assign_count)
        expected_lru = (
            list(range(assign_count, state["capacity"]))
            + list(range(assign_count))
        )

        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()
        actual = self.snapshot_external_lru_state(state)
        self.assertEqual(actual["last_req_ids"], [20260806])
        self.assertEqual(actual["current_slots"], [expected_current])
        self.assertEqual(actual["miss_count"], [assign_count])
        self.assertEqual(actual["miss_tokens"], [valid + [-1] * (state["topk"] - assign_count)])
        self.assertEqual(actual["miss_slots"], [list(range(assign_count)) + [-1] * (state["topk"] - assign_count)])
        self.assertEqual(actual["slot_to_token"], [expected_slot_to_token])
        self.assertEqual(actual["lru_slots"], [expected_lru])
        self.assertEqual(state["epochs"][0].cpu().item(), -2)
        print(
            "PRODUCTION_SIMT_LRU_RESET_FAST_PATH_MATRIX_PASS "
            f"external_state_sha256={self.external_lru_digest(actual)}",
            flush=True,
        )

    def test_lru_compact_production_shape_semantic_matrix(self):
        """production shape 综合语义矩阵，避免 mixed backend 因小 shape fallback。

        覆盖 stable-prefix invalidation、TopK first occurrence、重复 resident 的
        old-LRU last-wins、invalid LRU、稳定 hit/evict 顺序、miss assignment 和 epoch。
        """
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production A5 kernel package")

        topk, capacity, max_token = 2048, 4096, 4096
        topk_values = [5, 8, 5, -1, max_token, 9] + list(range(100, 2142))
        self.assertEqual(len(topk_values), topk)
        resident = [-1] * capacity
        resident[0] = 5
        resident[1] = 8
        resident[2] = 5  # Later old-LRU duplicate must win at first Top-K position.
        resident[3] = 9
        resident[4] = 4000  # Invalidated by stable_prefix_len.
        resident[5] = 777
        old_lru = list(range(capacity))
        old_lru[-1] = -7  # Preserve the serial invalid-LRU-entry behavior.

        # 独立 compatibility oracle 按公开五阶段语义推导，而非照抄 device 并行代码；
        # 这样能避免 kernel 和测试以同一种错误同时“通过”。
        expected_resident = resident.copy()
        for slot, token in enumerate(expected_resident):
            if 0 <= token < max_token and token >= 3500:
                expected_resident[slot] = -1
        first_pos = {}
        for pos, token in enumerate(topk_values):
            if 0 <= token < max_token and token not in first_pos:
                first_pos[token] = pos
        expected_current = [-1] * topk
        hits = []
        evictable = []
        for slot in old_lru:
            if not 0 <= slot < capacity:
                continue
            token = expected_resident[slot]
            if token in first_pos:
                expected_current[first_pos[token]] = slot
                hits.append(slot)
            else:
                evictable.append(slot)
        misses = [
            (token, pos)
            for pos, token in enumerate(topk_values)
            if 0 <= token < max_token and expected_current[pos] < 0
        ]
        assign_count = min(len(misses), len(evictable))
        expected_miss_tokens = [-1] * topk
        expected_miss_slots = [-1] * topk
        for miss_index, (token, pos) in enumerate(misses[:assign_count]):
            slot = evictable[miss_index]
            expected_resident[slot] = token
            expected_current[pos] = slot
            expected_miss_tokens[miss_index] = token
            expected_miss_slots[miss_index] = slot
        expected_lru = (
            evictable[assign_count:]
            + expected_miss_slots[:assign_count]
            + hits
            + [-1] * (capacity - len(evictable) - len(hits))
        )

        state = {
            "num_reqs": 1,
            "topk": topk,
            "capacity": capacity,
            "max_token": max_token,
            "req_ids": self.npu_tensor([99], torch.int64),
            "last_req_ids": self.npu_tensor([99], torch.int64),
            "topk_indices": self.npu_tensor([topk_values], torch.int32),
            "stable_prefix": self.npu_tensor([3500], torch.int32),
            "slot_to_token": self.npu_tensor([resident], torch.int32),
            "lru_slots": self.npu_tensor([old_lru], torch.int32),
            "current_slots": torch.full((1, topk), -1, dtype=torch.int32,
                                        device=self.device),
            "miss_count": torch.zeros(1, dtype=torch.int32, device=self.device),
            "miss_tokens": torch.full((1, topk), -1, dtype=torch.int32,
                                      device=self.device),
            "miss_slots": torch.full((1, topk), -1, dtype=torch.int32,
                                     device=self.device),
            "token_mark": torch.zeros((8, max_token), dtype=torch.int32,
                                      device=self.device),
            "token_pos": torch.full((8, max_token), -1, dtype=torch.int32,
                                    device=self.device),
            "epochs": torch.full((8,), 1 - (1 << 30), dtype=torch.int32,
                                 device=self.device),
        }
        self.assertEqual(self.run_production_lru(state), 0)
        torch.npu.synchronize()
        actual = self.snapshot_external_lru_state(state)
        self.assertEqual(actual["last_req_ids"], [99])
        self.assertEqual(actual["slot_to_token"], [expected_resident])
        self.assertEqual(actual["lru_slots"], [expected_lru])
        self.assertEqual(actual["current_slots"], [expected_current])
        self.assertEqual(actual["miss_count"], [assign_count])
        self.assertEqual(actual["miss_tokens"], [expected_miss_tokens])
        self.assertEqual(actual["miss_slots"], [expected_miss_slots])
        self.assertEqual(state["epochs"][0].cpu().item(), -2)
        print(
            "PRODUCTION_SIMT_LRU_SEMANTIC_MATRIX_PASS "
            f"external_state_sha256={self.external_lru_digest(actual)}",
            flush=True,
        )

    def test_lru_compact_duplicate_compat_semantics(self):
        """小尺寸回归：reset 后 duplicate TopK 的两个位置仍是两个独立 miss。"""
        if self.kernel_manifest.get("lru_resident_compact") != "mixed_ub_fused_v3":
            self.skipTest("install the production phased SIMT Compact")
        num_reqs, topk, capacity, max_token = 1, 2, 2, 32
        req_ids = self.npu_tensor([20], torch.int64)
        last_req_ids = self.npu_tensor([19], torch.int64)
        topk_indices = self.npu_tensor([[4, 4]], torch.int32)
        stable_prefix = self.npu_tensor([8], torch.int32)
        slot_to_token = self.npu_tensor([[1, 4]], torch.int32)
        lru_slots = self.npu_tensor([[0, 1]], torch.int32)
        current_slots = torch.full((1, topk), -1, dtype=torch.int32, device=self.device)
        miss_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        miss_tokens = torch.full((1, topk), -1, dtype=torch.int32, device=self.device)
        miss_slots = torch.full_like(miss_tokens, -1)
        token_mark = torch.zeros((8, max_token), dtype=torch.int32, device=self.device)
        token_pos = torch.full_like(token_mark, -1)
        epochs = torch.zeros(8, dtype=torch.int32, device=self.device)

        self.assertEqual(
            offload.lru_resident_compact(
                req_ids,
                last_req_ids,
                topk_indices,
                stable_prefix,
                slot_to_token,
                lru_slots,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                token_mark,
                token_pos,
                epochs,
                num_reqs,
                topk,
                capacity,
                max_token,
                self.device,
            ),
            0,
        )
        torch.npu.synchronize()
        self.assertEqual(current_slots.cpu().tolist(), [[0, 1]])
        self.assertEqual(miss_count.cpu().tolist(), [2])
        self.assertEqual(miss_tokens.cpu().tolist(), [[4, 4]])
        self.assertEqual(miss_slots.cpu().tolist(), [[0, 1]])
        self.assertEqual(slot_to_token.cpu().tolist(), [[4, 4]])

    # ======================================================================
    # 第二组：LRU -> ResidentAddrs -> SparseCopy 三算子真实流水线
    # ======================================================================

    def test_sparse_kv_three_operator_pipeline(self):
        """五种 hit ratio × 两个 copy 方向的 production-shape 端到端正确性。

        中间没有 CPU 伪造 descriptor；每个 kernel 的真实 device 输出直接交给下一个。
        所有 kernel 提交完成后只同步一次，再检查 task_count、地址和最终 K/V 字节。
        all-hit 特别验证 ``task_count=0`` 时 SparseCopy 是 no-op。
        """
        if os.getenv("MF_RUN_A5_SPARSE_KV_PIPELINE") != "1":
            self.skipTest("set MF_RUN_A5_SPARSE_KV_PIPELINE=1")
        if self.kernel_mode != "production":
            self.skipTest("pipeline validation requires the production kernel package")

        hit_cases = (
            ("fresh", None),
            ("hit25", 512),
            ("hit50", 1024),
            ("hit75", 1536),
            ("all_hit", 2048),
        )
        # 每个 subTest 都重新创建独立地址空间，便于报错时直接定位方向和 hit ratio。
        for direction in ("host_to_npu", "npu_to_host"):
            for label, hit_count in hit_cases:
                with self.subTest(direction=direction, workload=label):
                    case = self.new_sparse_kv_pipeline_case(hit_count, direction)
                    self.submit_sparse_kv_pipeline(case)
                    torch.npu.synchronize()
                    self.assert_sparse_kv_pipeline_result(case)
                    expected_misses = 2048 if hit_count is None else 2048 - hit_count
                    self.assertEqual(case["state"]["miss_count"].cpu().tolist(), [expected_misses])
                    self.assertEqual(case["task_count"].cpu().tolist(), [2 * expected_misses])

        print(
            "SPARSE_KV_THREE_OPERATOR_PIPELINE_PASS "
            "workloads=fresh,hit25,hit50,hit75,all_hit "
            "directions=host_to_npu,npu_to_host layout=all_k_then_all_v",
            flush=True,
        )

    def test_sparse_kv_pipeline_shuffled_state(self):
        """验证 fixed-seed shuffled TopK、离散 resident slot 和非 identity LRU。"""
        if os.getenv("MF_RUN_A5_SPARSE_KV_PIPELINE") != "1":
            self.skipTest("set MF_RUN_A5_SPARSE_KV_PIPELINE=1")
        if self.kernel_mode != "production":
            self.skipTest("pipeline validation requires the production kernel package")

        for direction in ("host_to_npu", "npu_to_host"):
            with self.subTest(direction=direction):
                case = self.new_sparse_kv_shuffled_pipeline_case(direction)
                self.submit_sparse_kv_pipeline(case)
                torch.npu.synchronize()
                self.assert_sparse_kv_pipeline_result(case)
                self.assertEqual(case["state"]["miss_count"].cpu().tolist(), [1024])
                self.assertEqual(case["task_count"].cpu().tolist(), [2048])

        print(
            "SPARSE_KV_PIPELINE_SHUFFLED_STATE_PASS "
            "seed=20260812 hits=1024 misses=1024 tasks=2048 "
            "resident_slots=shuffled lru=shuffled",
            flush=True,
        )

    def test_sparse_kv_pipeline_duplicate_and_invalid_topk(self):
        """让 duplicate miss 和 invalid token 穿过完整三算子链。

        TopK 前四项为 ``[4,4,-1,4096]``：前两个位置应分别占 slot 0/1，后两个
        无效且保持 -1。因此有效 miss=2046，最终必须产生 4092 个 K/V task。
        """
        if os.getenv("MF_RUN_A5_SPARSE_KV_PIPELINE") != "1":
            self.skipTest("set MF_RUN_A5_SPARSE_KV_PIPELINE=1")
        if self.kernel_mode != "production":
            self.skipTest("pipeline validation requires the production kernel package")

        topk_values = [4, 4, -1, 4096] + list(range(100, 2144))
        self.assertEqual(len(topk_values), 2048)
        expected_misses = 2046
        for direction in ("host_to_npu", "npu_to_host"):
            with self.subTest(direction=direction):
                case = self.new_sparse_kv_pipeline_case(
                    None, direction, topk_values=topk_values
                )
                self.submit_sparse_kv_pipeline(case)
                torch.npu.synchronize()
                self.assert_sparse_kv_pipeline_result(case)
                state = case["state"]
                self.assertEqual(state["miss_count"].cpu().tolist(), [expected_misses])
                self.assertEqual(case["task_count"].cpu().tolist(), [2 * expected_misses])
                self.assertEqual(state["current_slots"][0, :4].cpu().tolist(), [0, 1, -1, -1])
                self.assertEqual(state["miss_tokens"][0, :2].cpu().tolist(), [4, 4])

        print(
            "SPARSE_KV_PIPELINE_DUPLICATE_INVALID_PASS "
            f"misses={expected_misses} tasks={2 * expected_misses}",
            flush=True,
        )

    def test_sparse_kv_pipeline_latency(self):
        """对真实三算子链同时报告 stage-sync 与 end-to-end 两种性能口径。

        口径 A（stage）：每轮恢复相同初态后，分别测
        ``LRU+sync``、``Addr+sync``、``Copy+sync``，用于定位瓶颈。

        口径 B（pipeline）：恢复初态后连续提交三个 kernel，只在末尾同步一次，代表
        真实 stream pipeline。两种口径均排除 state reset、Host buffer 分配和断言。

        日志还输出 miss/task 数以及实际 copy bytes。默认 32B K + 48B V 仅用于轻量
        correctness；分析带宽时应设置真实 ``MF_A5_SPARSE_KV_TOKEN_BYTES_{K,V}``。
        """
        if os.getenv("MF_RUN_A5_SPARSE_KV_PIPELINE_PERF") != "1":
            self.skipTest("set MF_RUN_A5_SPARSE_KV_PIPELINE_PERF=1")
        if self.kernel_mode != "production":
            self.skipTest("pipeline benchmark requires the production kernel package")

        repeats = int(os.getenv("MF_A5_SPARSE_KV_PIPELINE_REPEATS", "100"))
        warmup = int(os.getenv("MF_A5_SPARSE_KV_PIPELINE_WARMUP", "10"))
        if repeats <= 0 or warmup < 0:
            self.fail("pipeline repeats must be positive and warmup must be non-negative")
        requested_directions = os.getenv(
            "MF_A5_SPARSE_KV_PIPELINE_DIRECTIONS", "host_to_npu,npu_to_host"
        )
        directions = tuple(item.strip() for item in requested_directions.split(",") if item.strip())
        hit_cases = (
            ("fresh", None),
            ("hit25", 512),
            ("hit50", 1024),
            ("hit75", 1536),
            ("all_hit", 2048),
        )

        for direction in directions:
            for label, hit_count in hit_cases:
                case = self.new_sparse_kv_pipeline_case(hit_count, direction)
                expected_misses = 2048 if hit_count is None else 2048 - hit_count
                task_count = 2 * expected_misses
                total_copy_bytes = expected_misses * (
                    case["token_bytes_k"] + case["token_bytes_v"]
                )

                # 一次完整 pipeline warmup 同时预热三个 public launch path。每轮仍
                # 恢复相同业务初态，warmup 的 state transition 不会泄漏到正式样本。
                for _ in range(warmup):
                    self.reset_sparse_kv_pipeline_case(case)
                    torch.npu.synchronize()
                    self.submit_sparse_kv_pipeline(case)
                    torch.npu.synchronize()

                lru_latencies = []
                addr_latencies = []
                copy_latencies = []
                for _ in range(repeats):
                    self.reset_sparse_kv_pipeline_case(case)
                    # reset 中包含异步 NPU fill/copy，必须在 timer 前同步完成。
                    torch.npu.synchronize()

                    started = time.perf_counter()
                    self.submit_sparse_kv_lru(case)
                    torch.npu.synchronize()
                    lru_latencies.append((time.perf_counter() - started) * 1000.0)

                    started = time.perf_counter()
                    self.submit_sparse_kv_addrs(case)
                    torch.npu.synchronize()
                    addr_latencies.append((time.perf_counter() - started) * 1000.0)

                    started = time.perf_counter()
                    self.submit_sparse_kv_copy(case)
                    torch.npu.synchronize()
                    copy_latencies.append((time.perf_counter() - started) * 1000.0)

                self.assertEqual(case["task_count"].cpu().tolist(), [task_count])
                lru_summary = self.latency_summary(lru_latencies)
                addr_summary = self.latency_summary(addr_latencies)
                copy_summary = self.latency_summary(copy_latencies)
                print(
                    "SPARSE_KV_PIPELINE_STAGE_PERF_PASS "
                    f"direction={direction} workload={label} warmup={warmup} repeats={repeats} "
                    f"hit_count={2048 - expected_misses} "
                    f"miss_count={expected_misses} task_count={task_count} "
                    f"token_bytes_k={case['token_bytes_k']} "
                    f"token_bytes_v={case['token_bytes_v']} "
                    f"total_copy_bytes={total_copy_bytes} "
                    f"{self.format_prefixed_latency_summary('lru', lru_summary)} "
                    f"{self.format_prefixed_latency_summary('addr', addr_summary)} "
                    f"{self.format_prefixed_latency_summary('copy', copy_summary)}",
                    flush=True,
                )

                # 端到端口径：三个 public API 之间没有 synchronize 或 CPU scalar read。
                pipeline_latencies = []
                for _ in range(repeats):
                    self.reset_sparse_kv_pipeline_case(case)
                    torch.npu.synchronize()  # 先排除测试状态恢复本身。
                    started = time.perf_counter()
                    self.submit_sparse_kv_pipeline(case)
                    torch.npu.synchronize()  # 三个算子提交后的唯一 pipeline 同步点。
                    pipeline_latencies.append((time.perf_counter() - started) * 1000.0)

                self.assertEqual(case["task_count"].cpu().tolist(), [task_count])
                self.assert_sparse_kv_pipeline_result(case)
                pipeline_summary = self.latency_summary(pipeline_latencies)
                print(
                    "SPARSE_KV_PIPELINE_END_TO_END_PERF_PASS "
                    f"direction={direction} workload={label} warmup={warmup} repeats={repeats} "
                    f"hit_count={2048 - expected_misses} "
                    f"miss_count={expected_misses} task_count={task_count} "
                    f"token_bytes_k={case['token_bytes_k']} "
                    f"token_bytes_v={case['token_bytes_v']} "
                    f"total_copy_bytes={total_copy_bytes} "
                    f"{self.format_prefixed_latency_summary('pipeline', pipeline_summary)}",
                    flush=True,
                )

                # stage_sum 包含三次独立 synchronize，不能解释成纯 device kernel sum。
                # delta 仅量化“真实一次同步口径”相对“诊断三次同步口径”的差异。
                stage_sync_sum = (
                    lru_summary["mean"] + addr_summary["mean"] + copy_summary["mean"]
                )
                print(
                    "SPARSE_KV_PIPELINE_TIMING_COMPARISON "
                    f"direction={direction} workload={label} "
                    f"stage_sync_sum_mean_ms={stage_sync_sum:.3f} "
                    f"pipeline_mean_ms={pipeline_summary['mean']:.3f} "
                    "delta_definition=pipeline_minus_stage_sync_sum "
                    f"pipeline_minus_stage_sync_sum_ms="
                    f"{pipeline_summary['mean'] - stage_sync_sum:.3f}",
                    flush=True,
                )

    # ======================================================================
    # 第三组：ResidentAddrs 与 SparseCopy 的独立边界/契约测试
    # ======================================================================

    def new_resident_addrs_matrix_case(self, count, include_invalid=True):
        """构造 production topk=2048 的可复现 Addr 独立测试输入。

        token、slot 和 block table 都不是 identity；可选的 invalid 交错同时覆盖
        token/slot 越界和 ``block_table=-1``。输出用不同 sentinel 填充，既能验证
        packed 前缀，也能发现 task_count 之外的意外写入。
        """
        topk, capacity, block_size, max_num_blocks = 2048, 4096, 128, 33
        rng = random.Random(20260812)
        tokens = [rng.randrange(0, 4096) for _ in range(topk)]
        slots = list(range(topk))
        rng.shuffle(slots)
        # duplicate 本身合法且不得去重；这些位置还覆盖 32/1024 两类边界。
        for index, token in ((0, 5), (31, 5), (32, 7), (1023, 9),
                             (1024, 9), (2047, 11)):
            tokens[index] = token
        block_table_values = list(range(max_num_blocks))
        rng.shuffle(block_table_values)
        if include_invalid:
            tokens[17] = -1
            tokens[513] = max_num_blocks * block_size
            slots[33] = -1
            slots[1025] = capacity
            invalid_block_id = 3
            block_table_values[invalid_block_id] = -1
            tokens[65] = invalid_block_id * block_size + 1

        return {
            "count": count,
            "topk": topk,
            "capacity": capacity,
            "block_size": block_size,
            "max_num_blocks": max_num_blocks,
            "token_bytes_k": 1024,
            "token_bytes_v": 128,
            "gvas_k_base": 1_000_000,
            "gvas_v_base": 2_000_000,
            "addr_k_base": 3_000_000,
            "addr_v_base": 4_000_000,
            "tokens_cpu": tokens,
            "slots_cpu": slots,
            "block_table_cpu": block_table_values,
            "miss_count": self.npu_tensor([count], torch.int32),
            "miss_tokens": self.npu_tensor([tokens], torch.int32),
            "miss_slots": self.npu_tensor([slots], torch.int32),
            "block_table": self.npu_tensor([block_table_values], torch.int32),
            "gvas": torch.full((2 * topk,), -101, dtype=torch.int64,
                               device=self.device),
            "addrs": torch.full((2 * topk,), -202, dtype=torch.int64,
                                device=self.device),
            "sizes": torch.full((2 * topk,), -303, dtype=torch.int32,
                                device=self.device),
            "task_count": self.npu_tensor([-1], torch.int32),
        }

    @staticmethod
    def expected_resident_addrs(case):
        """按 serial contract 计算 stable ``[all K][all V]`` oracle。"""
        valid = []
        count = max(0, min(case["count"], case["topk"]))
        for index in range(count):
            token = case["tokens_cpu"][index]
            slot = case["slots_cpu"][index]
            if token < 0 or slot < 0 or slot >= case["capacity"]:
                continue
            block_id, offset = divmod(token, case["block_size"])
            if block_id < 0 or block_id >= case["max_num_blocks"]:
                continue
            block_index = case["block_table_cpu"][block_id]
            if block_index < 0:
                continue
            valid.append((slot, block_index, offset))

        src_k, src_v, dst_k, dst_v = [], [], [], []
        for slot, block_index, offset in valid:
            src_k.append(
                case["gvas_k_base"]
                + block_index * case["block_size"] * case["token_bytes_k"]
                + offset * case["token_bytes_k"]
            )
            src_v.append(
                case["gvas_v_base"]
                + block_index * case["block_size"] * case["token_bytes_v"]
                + offset * case["token_bytes_v"]
            )
            dst_k.append(case["addr_k_base"] + slot * case["token_bytes_k"])
            dst_v.append(case["addr_v_base"] + slot * case["token_bytes_v"])
        total = len(valid)
        return {
            "gvas": src_k + src_v,
            "addrs": dst_k + dst_v,
            "sizes": ([case["token_bytes_k"]] * total
                      + [case["token_bytes_v"]] * total),
            "task_count": 2 * total,
            "valid_count": total,
        }

    def submit_resident_addrs_case(self, case):
        """只提交 Addr 公共 ABI；Host 启动层会按 shape 自动选择实现。

        当前公开精简版不再提供 ResidentAddrs 版本选择环境变量：
        ``num_reqs=1`` 且 ``0 < topk <= 2048`` 时自动使用最新的
        1024-lane mixed parallel kernel，其余 shape 自动走通用 serial SIMT
        fallback。测试始终调用同一个公共接口，正好覆盖真实生产分发路径。
        """
        self.assertEqual(
            offload.compute_lru_resident_addrs(
                case["miss_count"], case["miss_tokens"], case["miss_slots"],
                case["block_table"], case["gvas"], case["addrs"],
                case["sizes"], case["task_count"], case["block_size"],
                case["token_bytes_k"], case["token_bytes_v"],
                case["gvas_k_base"], case["gvas_v_base"],
                case["addr_k_base"], case["addr_v_base"], case["capacity"],
                1, case["topk"], case["max_num_blocks"], self.device,
            ),
            0,
        )

    def assert_resident_addrs_case(self, case):
        """逐元素比对 oracle，并确认 packed 区域之后的 sentinel 未被写。"""
        expected = self.expected_resident_addrs(case)
        actual = self.resident_addrs_actual_payload(case)
        tasks = actual["task_count"]
        self.assertEqual(tasks, expected["task_count"])
        self.assertEqual(actual["gvas"][:tasks], expected["gvas"])
        self.assertEqual(actual["addrs"][:tasks], expected["addrs"])
        self.assertEqual(actual["sizes"][:tasks], expected["sizes"])
        self.assertTrue(all(value == -101 for value in actual["gvas"][tasks:]))
        self.assertTrue(all(value == -202 for value in actual["addrs"][tasks:]))
        self.assertTrue(all(value == -303 for value in actual["sizes"][tasks:]))
        return expected

    @staticmethod
    def resident_addrs_actual_payload(case):
        """返回真实 device 输出，供断言和跨独立进程 digest 共同使用。

        整个 descriptor buffer 都纳入 payload，因此 task_count 之外的 sentinel
        也会参与 ``actual_sha256``，不仅能证明 packed prefix 相同，也能发现
        serial/parallel 任一后端对尾部的意外写入。
        """
        return {
            "task_count": int(case["task_count"].cpu().item()),
            "gvas": case["gvas"].cpu().tolist(),
            "addrs": case["addrs"].cpu().tolist(),
            "sizes": case["sizes"].cpu().tolist(),
        }

    def test_resident_addresses_parallel_boundary_matrix(self):
        """跨 warp/tile 边界验证 stable filter、duplicate 和 invalid contract。

        这些 case 都满足最新 parallel kernel 的 shape gate，因此应由 Host 层
        自动分发到 mixed parallel 实现。每个 case 都逐元素对 CPU oracle 做完整
        断言；最终 digest 用来方便保存和比较整组真实 device 输出。
        """
        if os.getenv("MF_RUN_A5_RESIDENT_ADDRS_MATRIX") != "1":
            self.skipTest("set MF_RUN_A5_RESIDENT_ADDRS_MATRIX=1")
        counts = (0, 1, 31, 32, 33, 511, 512, 513, 1023, 1024,
                  1025, 1536, 2047, 2048)
        oracle_digest = hashlib.sha256()
        actual_digest = hashlib.sha256()
        for count in counts:
            case = self.new_resident_addrs_matrix_case(count)
            self.submit_resident_addrs_case(case)
            torch.npu.synchronize()
            expected = self.assert_resident_addrs_case(case)
            actual = self.resident_addrs_actual_payload(case)
            oracle_digest.update(json.dumps(
                expected, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
            actual_digest.update(json.dumps(
                actual, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
        print(
            "RESIDENT_ADDRS_BOUNDARY_MATRIX_PASS "
            "backend=mixed_simt_parallel "
            f"cases={len(counts)} "
            f"oracle_sha256={oracle_digest.hexdigest()} "
            f"actual_sha256={actual_digest.hexdigest()}",
            flush=True,
        )

    def test_resident_addresses_parallel_unsupported_shape_fallback(self):
        """``num_reqs=2`` 必须安全 fallback，并保留 serial 跨 row 布局。

        当前精简版按 shape 自动分发：``num_reqs=2`` 不满足 mixed parallel
        专用条件，因此必须自动进入通用 serial SIMT。该测试不把 fallback 日志
        当 correctness oracle，而是逐元素验证回退后的全局 stable
        ``[all K][all V]`` descriptor 和尾部 sentinel。
        """
        if os.getenv("MF_RUN_A5_RESIDENT_ADDRS_FALLBACK") != "1":
            self.skipTest("set MF_RUN_A5_RESIDENT_ADDRS_FALLBACK=1")
        miss_count = self.npu_tensor([2, 2], torch.int32)
        miss_tokens = self.npu_tensor([[5, -1], [9, 6]], torch.int32)
        miss_slots = self.npu_tensor([[2, -1], [1, 3]], torch.int32)
        block_table = self.npu_tensor(
            [[4, 7, 0, 0], [3, 1, 6, 2]], torch.int32
        )
        gvas = torch.full((8,), -101, dtype=torch.int64, device=self.device)
        addrs = torch.full((8,), -202, dtype=torch.int64, device=self.device)
        sizes = torch.full((8,), -303, dtype=torch.int32, device=self.device)
        task_count = self.npu_tensor([-1], torch.int32)

        self.assertEqual(offload.compute_lru_resident_addrs(
            miss_count, miss_tokens, miss_slots, block_table,
            gvas, addrs, sizes, task_count,
            4, 8, 16, 1000, 5000, 10000, 20000,
            4, 2, 2, 4, self.device,
        ), 0)
        torch.npu.synchronize()

        self.assertEqual(task_count.cpu().tolist(), [6])
        self.assertEqual(
            gvas.cpu().tolist(),
            [1232, 1200, 1048, 5464, 5400, 5096, -101, -101],
        )
        self.assertEqual(
            addrs.cpu().tolist(),
            [10016, 10040, 10056, 20032, 20080, 20112, -202, -202],
        )
        self.assertEqual(
            sizes.cpu().tolist(), [8, 8, 8, 16, 16, 16, -303, -303]
        )
        print(
            "RESIDENT_ADDRS_UNSUPPORTED_SHAPE_FALLBACK_PASS "
            "backend=mixed_simt_parallel num_reqs=2 fallback=simt_serial",
            flush=True,
        )

    def test_resident_addresses_production_performance(self):
        """production topk 下测 0/512/1024/1536/2048 miss submit+sync。"""
        if os.getenv("MF_RUN_A5_RESIDENT_ADDRS_PERF") != "1":
            self.skipTest("set MF_RUN_A5_RESIDENT_ADDRS_PERF=1")
        warmup = int(os.getenv("MF_A5_RESIDENT_ADDRS_WARMUP", "10"))
        repeats = int(os.getenv("MF_A5_RESIDENT_ADDRS_REPEATS", "1000"))
        if warmup < 0 or repeats <= 0:
            self.fail("ResidentAddrs warmup must be >=0 and repeats must be >0")
        backend = "mixed_simt_parallel"
        for count in (0, 512, 1024, 1536, 2048):
            case = self.new_resident_addrs_matrix_case(
                count, include_invalid=False
            )
            expected = self.expected_resident_addrs(case)
            for _ in range(warmup):
                self.submit_resident_addrs_case(case)
                torch.npu.synchronize()
            latencies = []
            for _ in range(repeats):
                started = time.perf_counter()
                self.submit_resident_addrs_case(case)
                torch.npu.synchronize()
                latencies.append((time.perf_counter() - started) * 1000.0)
            self.assert_resident_addrs_case(case)
            summary = self.latency_summary(latencies)
            print(
                "RESIDENT_ADDRS_PERF_PASS "
                f"backend={backend} threads="
                "1024 "
                f"miss_count={count} valid_count={expected['valid_count']} "
                f"task_count={expected['task_count']} "
                f"{self.format_latency_summary(summary)}",
                flush=True,
            )

    def test_resident_addresses(self):
        """小尺寸手算 ResidentAddrs 的 K/V 地址、长度和偶数 task_count。

        token=5、block_size=4，因此 block_id=1、block 内 offset=1；block_table[1]=7。
        断言中的 1232/5464/10016/20032 均由该公式直接推导。
        """
        miss_count = self.npu_tensor([1], torch.int32)
        miss_tokens = self.npu_tensor([[5, -1]], torch.int32)
        miss_slots = self.npu_tensor([[2, -1]], torch.int32)
        block_table = self.npu_tensor([[4, 7, 0, 0]], torch.int32)
        gvas = torch.zeros(4, dtype=torch.int64, device=self.device)
        addrs = torch.zeros(4, dtype=torch.int64, device=self.device)
        sizes = torch.zeros(4, dtype=torch.int32, device=self.device)
        task_count = torch.zeros(1, dtype=torch.int32, device=self.device)

        self.assertEqual(offload.compute_lru_resident_addrs(
            miss_count, miss_tokens, miss_slots, block_table, gvas, addrs, sizes, task_count,
            4, 8, 16, 1000, 5000, 10000, 20000, 4, 1, 2, 4, self.device), 0)
        torch.npu.synchronize()

        self.assertEqual(task_count.cpu().tolist(), [2])
        self.assertEqual(gvas.cpu().tolist()[:2], [1232, 5464])
        self.assertEqual(addrs.cpu().tolist()[:2], [10016, 20032])
        self.assertEqual(sizes.cpu().tolist()[:2], [8, 16])

    def test_sparse_copy_kv_pair_contract(self):
        """NPU→NPU smoke：确认 SparseCopy 只接收偶数 ``[all K][all V]`` task。

        四个不同长度覆盖非对齐小块；这里不把奇数 task 当合法输入，因为生产上游
        ResidentAddrs 永远生成 ``2 * valid_miss_count``。
        """
        sources = [
            self.npu_tensor(list(range(5)), torch.uint8),
            self.npu_tensor(list(range(8)), torch.uint8),
            self.npu_tensor(list(range(3)), torch.uint8),
            self.npu_tensor(list(range(7)), torch.uint8),
        ]
        destinations = [torch.zeros_like(item) for item in sources]
        src_ptrs = self.npu_tensor([item.data_ptr() for item in sources], torch.int64)
        dst_ptrs = self.npu_tensor([item.data_ptr() for item in destinations], torch.int64)
        lengths = self.npu_tensor([item.numel() for item in sources], torch.int32)
        task_count = self.npu_tensor([len(sources)], torch.int32)
        self.assertEqual(len(sources) % 2, 0)

        self.assertEqual(offload.sparse_copy(src_ptrs, dst_ptrs, lengths, task_count, self.device), 0)
        torch.npu.synchronize()
        for source, destination in zip(sources, destinations):
            self.assertTrue(torch.equal(source.cpu(), destination.cpu()))

    def test_sparse_copy_registered_host_bidirectional_matrix(self):
        """十种长度在 NPU↔registered Host 两个方向各重复三轮。

        ``offload.empty`` 分配可注册 Host 内存，``get_device_address`` 获得 NPU DVA。
        公开生产版固定使用已经验证的 AIV/DataCopyPad SparseCopy。
        """
        sizes = [1, 7, 8, 31, 32, 33, 128, 257, 1024, 2049]
        repeats = 3
        hosts = [offload.empty([size], dtype=torch.uint8).zero_() for size in sizes]
        host_device_addresses = [int(offload.get_device_address(host)) for host in hosts]
        self.assertTrue(all(address != 0 for address in host_device_addresses))

        lengths = self.npu_tensor(sizes, torch.int32)
        task_count = self.npu_tensor([len(sizes)], torch.int32)
        for repeat in range(repeats):
            sources = [
                (torch.arange(size, dtype=torch.int32, device=self.device) + 17 * repeat + index)
                .remainder(251)
                .to(torch.uint8)
                for index, size in enumerate(sizes)
            ]
            destinations = [torch.zeros_like(source) for source in sources]
            for host in hosts:
                host.zero_()

            # 一次 submission 同时覆盖全部长度；先同步再 CPU 比较，避免读取异步结果。
            src_ptrs = self.npu_tensor([source.data_ptr() for source in sources], torch.int64)
            dst_ptrs = self.npu_tensor(host_device_addresses, torch.int64)
            self.assertEqual(
                offload.sparse_copy(src_ptrs, dst_ptrs, lengths, task_count, self.device),
                0,
            )
            torch.npu.synchronize()
            for size, host, source in zip(sizes, hosts, sources):
                with self.subTest(direction="npu_to_host", repeat=repeat, size=size):
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(host, source.cpu()))

            src_ptrs = self.npu_tensor(host_device_addresses, torch.int64)
            dst_ptrs = self.npu_tensor(
                [destination.data_ptr() for destination in destinations], torch.int64
            )
            self.assertEqual(
                offload.sparse_copy(src_ptrs, dst_ptrs, lengths, task_count, self.device),
                0,
            )
            torch.npu.synchronize()
            for size, destination, host in zip(sizes, destinations, hosts):
                with self.subTest(direction="host_to_npu", repeat=repeat, size=size):
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(destination.cpu(), host))

    def test_sparse_copy_registered_host_unaligned_addresses(self):
        """验证非零起始 offset、非对齐长度以及 copy 区域前后的 guard bytes。

        每个 case 分配 ``offset + size + guard``，只把中间窗口地址交给 SparseCopy。
        除目标窗口必须逐字节相等外，头尾 guard 必须保持 0，用于发现越界搬运。
        同一组 case 同时覆盖 NPU→Host 和 Host→NPU。
        """
        cases = [(1, 8), (1, 32), (3, 128), (5, 257)]
        repeats = 3
        guard_bytes = 5
        hosts = [
            offload.empty([offset + size + guard_bytes], dtype=torch.uint8).zero_()
            for offset, size in cases
        ]
        host_device_addresses = [int(offload.get_device_address(host)) for host in hosts]
        self.assertTrue(all(address != 0 for address in host_device_addresses))
        lengths = self.npu_tensor([size for _, size in cases], torch.int32)
        task_count = self.npu_tensor([len(cases)], torch.int32)

        for repeat in range(repeats):
            sources = [
                (
                    torch.arange(offset + size + guard_bytes, dtype=torch.int32,
                                 device=self.device)
                    + 29 * repeat
                    + index
                ).remainder(251).to(torch.uint8)
                for index, (offset, size) in enumerate(cases)
            ]
            destinations = [torch.zeros_like(source) for source in sources]
            for host in hosts:
                host.zero_()

            src_ptrs = self.npu_tensor(
                [source.data_ptr() + offset for source, (offset, _) in zip(sources, cases)],
                torch.int64,
            )
            dst_ptrs = self.npu_tensor(
                [address + offset for address, (offset, _) in zip(host_device_addresses, cases)],
                torch.int64,
            )
            self.assertEqual(
                offload.sparse_copy(src_ptrs, dst_ptrs, lengths, task_count, self.device),
                0,
            )
            torch.npu.synchronize()
            for (offset, size), host, source in zip(cases, hosts, sources):
                with self.subTest(direction="npu_to_host", repeat=repeat,
                                  offset=offset, size=size):
                    torch.npu.synchronize()
                    expected = source.cpu()[offset:offset + size]
                    self.assertTrue(torch.equal(host[offset:offset + size], expected))
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(host[:offset], torch.zeros_like(host[:offset])))
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(
                        host[offset + size:], torch.zeros_like(host[offset + size:])
                    ))

            src_ptrs = self.npu_tensor(
                [address + offset for address, (offset, _) in zip(host_device_addresses, cases)],
                torch.int64,
            )
            dst_ptrs = self.npu_tensor(
                [destination.data_ptr() + offset
                 for destination, (offset, _) in zip(destinations, cases)],
                torch.int64,
            )
            self.assertEqual(
                offload.sparse_copy(src_ptrs, dst_ptrs, lengths, task_count, self.device),
                0,
            )
            torch.npu.synchronize()
            for (offset, size), destination, host in zip(cases, destinations, hosts):
                with self.subTest(direction="host_to_npu", repeat=repeat,
                                  offset=offset, size=size):
                    torch.npu.synchronize()
                    actual = destination.cpu()
                    self.assertTrue(
                        torch.equal(actual[offset:offset + size], host[offset:offset + size])
                    )
                    torch.npu.synchronize()
                    self.assertTrue(
                        torch.equal(actual[:offset], torch.zeros_like(actual[:offset]))
                    )
                    torch.npu.synchronize()
                    self.assertTrue(torch.equal(
                        actual[offset + size:], torch.zeros_like(actual[offset + size:])
                    ))


if __name__ == "__main__":
    unittest.main()
