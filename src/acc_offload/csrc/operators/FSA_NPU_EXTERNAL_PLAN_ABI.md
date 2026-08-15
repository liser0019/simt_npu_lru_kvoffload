# FSA NPU External Plan ABI V1

`SparseKvPlanFsaRuntime` 在 NPU 上直接生成 fused sparse attention 消费的
`int16` 计划。计划不会回传 CPU，也不会经过 legacy membership map。

## 布局

```text
dtype            int16
device           NPU
shape            [logical_rows, row_stride]
alignment        16 int16（32 bytes）
control_offset   align_up(topk, 16)
row_stride       control_offset + 16
```

每行 `[0, topk)` 是 position-major encoded values；
`[topk, control_offset)` 是对齐 padding；`[control_offset,
control_offset + 8)` 是 control；其余 8 个 `int16` 保留并清零。

encoded value：

- `0`：无效 token 或未分配 slot；
- `slot + 1`：resident hit，或已由 current-token injection 写入的 miss；
- `-(slot + 1)`：fused kernel 需要从 full Host KV 搬入的普通 miss。

control：

| index | value | meaning |
|---:|---:|---|
| 0 | `0x5A4D` | plan row ready |
| 1 | `0x5A45` | external NPU plan ready |
| 2 | `topk` | logical plan length |
| 3 | `0` | plan values offset |
| 4 | `physical_row` | persistent selection row |
| 5 | `0x5A44` | direct physical-row layout |
| 6 | `capacity` | selection row stride |
| 7 | `0x5A56` | K/V paired copy mode |

Plan values和普通 control 字段先写，ready marker 最后发布。Plan kernel 与 fused
attention 必须在同一 NPU stream 顺序提交；ABI 不允许通过 `.cpu()`、Host callback
或 synchronize 桥接。

`topk=2048` 时：`control_offset=2048`、`row_stride=2064 int16`、
`bytes_per_row=4128`。

## 行域与状态域

- TopK、visible/stable lens、miss 输出和 plan 按 logical row 索引；
- `last_req_ids/slot_to_token/lru_slots` 按 persistent physical row 索引；
- device mapping kernel先复用相同 request id 的 physical row，再使用空 row，最后
  使用第一个未占用 row；
- FSA 模式只管理 `[0, capacity-1)`，最后一个 slot 保留给 current token。

普通 `SparseKvLoadRuntime` 不使用本 ABI，且不会被本协议收窄。
