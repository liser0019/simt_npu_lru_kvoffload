# local_dram_offload

## 场景
单机多卡 KV offload 实例：每 rank 在本地 DRAM 上创建 1GB offload 池，存放 KV cache，并通过 `sparse_copy` 将 host 侧 KV 批量拷贝到 NPU，完成一次写入与读回校验。

## 目标
验证 ACC OFFLOAD 基础生命周期与稀疏拷贝数据通路在多卡并发下是否工作正常。

## 使用能力
- `mf.set_log_level(level)`
- `offload.initialize(config: OffloadConfig)` / `offload.uninitialize()`
- `OffloadConfig`（`device_id`、`reserve_size`、`alloc_size`）
- `offload.empty(sizes, dtype, pin_memory)`（在 host DRAM 池分配 KV tensor）
- `offload.sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId)`（批量稀疏 H2D 拷贝）

## 规模建议
- world_size=4（4 进程 4 卡，spawn 启动）
- 每 rank offload 池 reserve_size=alloc_size=1GB
- KV 维度：K_DIM=512、V_DIM=64，dtype=bfloat16
- 每 rank tokens=4×2048=8192 组 KV

## 必要条件
- 单机至少 4 张可用 NPU（device id 0/1/2/3）
- 已安装 **`memfabric_hybrid`**、`torch`、`torch_npu`、`numpy`

## 验收标准
- 各 rank 的 `offload.initialize` 与 `offload.sparse_copy` 均返回 0
- 拷贝完成后 NPU 侧 dst tensor 之和为 0（host 池零值已覆盖 device 上的 ones）
- 4 个子进程全部正常退出（exitcode=0），打印 `local_dram_offload: all ranks OK`

## 运行（Python）

```bash
cd examples/kv_offload/local_dram_offload
python3 local_dram_offload.py
```

成功时打印 `local_dram_offload: all ranks OK`。

## Q&A
