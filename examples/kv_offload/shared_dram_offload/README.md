# shared_dram_offload

## 场景
单机多卡 KV offload 实例（共享 DRAM 池）：4 个 rank 共享同一份 1GB offload 池——由 rank 0 贡献物理内存，其余 rank 共享访问该池。各 rank 通过 `sparse_copy` 将 host 侧 KV 批量拷贝到各自 NPU，完成一次写入与读回校验。

## 目标
验证 ACC OFFLOAD 在 `SHARED` 场景下的多卡组网、共享池建立与稀疏拷贝数据通路是否工作正常。

## 使用能力
- `mf.set_log_level(level)`
- `offload.initialize(config: OffloadConfig)` / `offload.uninitialize()`
- `OffloadConfig`（`device_id`、`reserve_size`、`alloc_size`、`world_size`、`rank_id`、`scene`）
- `offload.Scene` 枚举（`LOCAL`、`SHARED`）
- `offload.empty(sizes, dtype, pin_memory)`（在共享 DRAM 池分配 KV tensor）
- `offload.sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId)`（批量稀疏 H2D 拷贝）

## 规模建议
- world_size=4（4 进程 4 卡，spawn 启动）
- 共享 offload 池 reserve_size=1GB，rank 0 alloc_size=1GB 贡献物理内存，其余 rank alloc_size=0
- KV 维度：K_DIM=512、V_DIM=64，dtype=bfloat16
- 每 rank tokens=4×2048=8192 组 KV

## 必要条件
- 单机至少 4 张可用 NPU（device id 0/1/2/3）
- 已安装 **`memfabric_hybrid`**、`torch`、`torch_npu`、`numpy`
- `SHARED` 场景底层依赖 smem_bm 组网，`offload.initialize` 与 `offload.uninitialize` 均为**集合操作**，需所有 rank 同步进入

## 验收标准
- 各 rank 的 `offload.initialize` 与 `offload.sparse_copy` 均返回 0
- 拷贝完成后 NPU 侧 dst tensor 之和为 0（host 池零值已覆盖 device 上的 ones）
- 4 个子进程全部正常退出（exitcode=0），打印 `shared_dram_offload: all ranks OK`

## 运行（Python）

```bash
cd examples/kv_offload/shared_dram_offload
python3 shared_dram_offload.py
```

成功时打印 `shared_dram_offload: all ranks OK`。

## 与 local_dram_offload 的差异
- `scene` 设为 `offload.Scene.SHARED`，并补充 `world_size`、`rank_id` 配置
- `alloc_size` 仅 rank 0 传实际值（贡献物理内存），其余 rank 传 0；`reserve_size` 各 rank 保持一致
- `offload.initialize` / `offload.uninitialize` 是集合操作，用例通过两个 `mp.Barrier` 保证 4 个 rank 同步进入与退出，避免组网挂死

## Q&A
