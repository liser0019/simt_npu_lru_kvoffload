# SparseKvLoadRuntime 两阶段设计

## 1. 目标

旧热路径包含三个算子：

```text
LruResidentCompact
        ↓ miss token / slot
ComputeLruResidentAddrs
        ↓ src / dst / size descriptors
SparseCopy
        ↓
resident KV
```

`ComputeLruResidentAddrs` 只把 `(token, slot)` 翻译成地址描述符，描述符随后
立即被 `SparseCopy` 读回。新路径把它改为一个业务 API、两个同流 kernel：

```text
sparse_kv_load_runtime(...)
        │
        ├─ SparseKvPlanRuntime
        │      TopK + resident/LRU → compact miss plan
        │
        └─ SparseKvTransferRuntime
               miss plan → 现场算地址 → DataCopyPad H2D
```

两个 kernel 之间没有 Host scalar read、callback 或 synchronize；同一 NPU stream
的提交顺序构成 Plan → Transfer 的数据依赖。

旧三算子暂时保留用于 A/B、回滚和真机 oracle，不参与新 API 的正常分发。

## 2. 语义基线

Plan 以 vLLM-Ascend CPU `lru_resident_compact` 为语义真值：

- request id 改变时，清空 resident token，并把 LRU 重置为 identity；
- `stable_prefix_len` 之外的 speculative resident token 先失效；
- TopK hash 保存 token 的第一次出现位置；
- resident 命中写入第一次 TopK 位置；若 resident 状态错误地含有重复 token，
  old-LRU 中靠后的 slot 最终覆盖 `current_slots[first_position]`，但全部命中
  slot 都保留在 LRU hit 段；
- TopK 后续重复位置仍可能成为独立 miss，不做全局去重；
- hit、evictable 和 miss 均保持原顺序；
- miss 使用最老的 evictable slot；
- 新 LRU 顺序为 `未使用 evictable + 新分配 miss + hit`。

非法 token、非法 LRU slot 不参与 Plan；`max_token` 只用于 token domain 校验，
不参与 workspace 大小计算。

## 3. Plan kernel

外层入口为 Ascend mixed kernel：

```cpp
extern "C" __global__ __vector__ void SparseKvPlanRuntimeKernel(...)
```

每个 outer block 调用：

```cpp
asc_vf_call<SparseKvPlanRuntimeVf>(dim3(1024), ...)
```

一次 Host launch 最多启用 32 个 outer block。每个 block 以 block-stride 处理
request 行，因此 `num_reqs` 是 runtime 值，而不是 Host 逐行 launch。

每个 request 内由 1024 个 SIMT lane 处理，`topk` 和 `capacity` 都按 1024
元素 tile 遍历。尾 tile 的 inactive lane 只贡献 flag=0，仍参与全部
`asc_syncthreads()`，避免 barrier divergence。

### 3.1 runtime hash

每行 hash capacity 为：

```text
next_power_of_two(max(32, 2 * topk))
```

负载因子不超过 0.5。hash 位于 GM，使用线性 probing：

- `asc_atomic_cas` 发布 token key；
- `asc_atomic_min` 保存第一次 TopK position；
- `asc_atomic_max` 选择 old-LRU 中最后一个 resident owner。

这些操作位于同一 request block 内；阶段之间用 `asc_syncthreads()` 保证
同 block lane 的先前 GM 操作完成并可见。这里没有跨 block 共享 hash。

### 3.2 stable scan

`BlockExclusiveScan1024` 使用 32 个 32-lane warp：

1. 每个 warp 用 `asc_shfl_up` 做 inclusive scan；
2. lane 31 把 warp total 写入 288-byte UB scratch；
3. warp 0 扫描 32 个 warp total；
4. 得到每 lane 的 block exclusive rank。

tile 之间用 carry 累加，因此同一算法支持任意 runtime `topk/capacity`，
没有 `topk <= 2048` 之类分支。

### 3.3 workspace

每 request 的 int32 workspace 为：

```text
hash_keys[hash_capacity]
hash_first_position[hash_capacity]
hash_resident_owner[hash_capacity]
evictable_slots[capacity]
hit_slots[capacity]
miss_positions[topk]
```

所以：

```text
workspace = O(num_reqs * (topk + capacity))
```

固定 `num_reqs/topk/capacity` 时，`max_token=4096/128K/1M` 返回完全相同的
workspace size。Plan 显式 UB 只有 288-byte scan scratch，不随 shape 增长。

## 4. Transfer kernel

Transfer 继续使用已验证的 AIV/DataCopyPad registered-Host-DVA 路径，启动
32 个 AIV block。它不接收 `gvas/addr/size/task_count`。

每个 request 中，AIV `core` 处理：

```text
miss index = core, core + 32, core + 64, ...
```

每个 miss 现场读取 token/slot 和 request-strided block table，计算：

```text
source = host_DVA_base
       + block_index * block_size * token_bytes
       + offset_in_block * token_bytes

destination = device_base
            + (request * capacity + slot) * token_bytes
```

随后分别对 K、V 执行 GM → UB → GM `DataCopyPad`。这样保留 32-AIV copy
并行度，同时消除 descriptor 写入、再次读取以及 multi-row global prefix。

## 5. API 与限制

公开 API：

```text
get_sparse_kv_plan_workspace_size(num_reqs, topk, capacity)
sparse_kv_load_runtime(...)
```

业务 shape 全部 runtime。当前 ABI 的实际限制是：

- token/slot/hash position 为 int32；`max_token <= INT32_MAX`；
- 为维持 `2 * topk` hash target，`topk <= INT32_MAX / 2`；
- `capacity <= INT32_MAX`；
- workspace、tensor shape 与地址乘法必须无整数溢出；
- Transfer 当前是 registered Host DVA → NPU resident buffer（load/H2D）。

非法 Host 参数返回错误，不分发到旧 kernel。Transfer 对 device 侧非法 miss、
slot 或 block-table entry 做防御性跳过。

## 6. 验证状态

本分支已提供：

- 独立 CPU Plan oracle（含 duplicate/reset/stable-prefix/LRU 语义）；
- runtime shape 和 fixed-seed random property tests；
- descriptor-free Transfer CPU payload oracle；
- workspace 不依赖 max_token 的测试；
- opt-in A5 registered-DVA Plan+Transfer integration test；
- source contract：Plan 在 Transfer 之前同流提交且无中间 synchronize。

当前开发机没有 CANN 9.1、Bisheng 和 A5，因此：

```text
CPU/source tests: PASS
Bisheng build: NEEDS_A5_BUILD
A5 correctness/runtime: NEEDS_A5_RUNTIME_TEST
A5 performance: NEEDS_A5_RUNTIME_TEST
vLLM manager migration: NEEDS_VLLM_INTEGRATION（位于独立仓库）
```

A5 首轮必须确认 GM atomic API 编译、1024-lane VF barrier、register spill、
registered Host DVA DataCopyPad 和 32-AIV 负载。未得到这些证据前，不应删除
旧三算子实现。
