#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import multiprocessing as mp
import torch
import torch.distributed as dist
import torch_npu
import memfabric_hybrid as mf
from memfabric_hybrid import offload


ONE_GIB = 1 << 30
WORLD_SIZE = 4
RANK_0, DEVICE_0 = 0, 0
RANK_1, DEVICE_1 = 1, 1
RANK_2, DEVICE_2 = 2, 2
RANK_3, DEVICE_3 = 3, 3
K_DIM = 512
V_DIM = 64


def _rank_main(rank_id: int, device_id: int, sync: mp.Barrier):
    mf.set_log_level(3)
    torch.npu.set_device(device_id)

    config = offload.OffloadConfig()
    config.device_id = device_id
    config.reserve_size = ONE_GIB
    config.alloc_size = ONE_GIB if rank_id == 0 else 0
    config.world_size = WORLD_SIZE
    config.rank_id = rank_id
    config.scene = offload.Scene.SHARED
    assert offload.initialize(config) == 0, f"rank_id:{rank_id} offload.initialize failed"

    data = {
        'cpu': {'keys': [], 'values': [], 'key_ptrs': [], 'value_ptrs': []},
        'npu': {'keys': [], 'values': [], 'key_ptrs': [], 'value_ptrs': []},
        'len': {'keys': [], 'values': []}
    }

    tokens = 4 * 2048  # batch * tokens_per_req
    for _ in range(tokens):
        if rank_id == 0:
            cpu_key = offload.empty([K_DIM, 1], dtype=torch.bfloat16).zero_()
            cpu_value = offload.empty([V_DIM, 1], dtype=torch.bfloat16).zero_()
        else:
            cpu_key = torch.ones([K_DIM, 1], dtype=torch.bfloat16)
            cpu_value = torch.ones([V_DIM, 1], dtype=torch.bfloat16)
        npu_key = torch.ones(K_DIM, dtype=torch.bfloat16).npu()
        npu_value = torch.ones(V_DIM, dtype=torch.bfloat16).npu()

        data['cpu']['keys'].append(cpu_key)
        data['cpu']['values'].append(cpu_value)
        data['cpu']['key_ptrs'].append(cpu_key.data_ptr())
        data['cpu']['value_ptrs'].append(cpu_value.data_ptr())

        data['npu']['keys'].append(npu_key)
        data['npu']['values'].append(npu_value)
        data['npu']['key_ptrs'].append(npu_key.data_ptr())
        data['npu']['value_ptrs'].append(npu_value.data_ptr())

        data['len']['keys'].append(cpu_key.numel() * torch.bfloat16.itemsize)
        data['len']['values'].append(cpu_value.numel() * torch.bfloat16.itemsize)

    size = len(data['cpu']['key_ptrs'] + data['cpu']['value_ptrs'])
    src_ptrs = torch.tensor(data['cpu']['key_ptrs'] + data['cpu']['value_ptrs'], dtype=torch.int64).npu()
    dst_ptrs = torch.tensor(data['npu']['key_ptrs'] + data['npu']['value_ptrs'], dtype=torch.int64).npu()
    len_ptrs = torch.tensor(data['len']['keys'] + data['len']['values'], dtype=torch.int32).npu()
    size_ptr = torch.tensor(size, dtype=torch.int32).npu()
    device = data['npu']['keys'][0].device

    group = dist.init_process_group("hccl", init_method='tcp://127.0.0.1:23456', rank=rank_id, world_size=WORLD_SIZE)
    dist.broadcast(src_ptrs, src=0)

    assert offload.sparse_copy(src_ptrs, dst_ptrs, len_ptrs, size_ptr, device) == 0, "offload.sparse_copy failed"
    torch.npu.synchronize()

    dst_tensors = data['npu']['keys'] + data['npu']['values']
    for dst_tensor in dst_tensors:
        dst_sum = dst_tensor.sum().item()
        assert dst_sum == 0, f"rank_id:{rank_id} dst tensor values not correct: {dst_tensor}"

    offload.uninitialize()
    sync.wait()


def main():
    mp.set_start_method("spawn", force=True)
    sync = mp.Barrier(WORLD_SIZE)

    p0 = mp.Process(target=_rank_main, args=(RANK_0, DEVICE_0, sync))
    p1 = mp.Process(target=_rank_main, args=(RANK_1, DEVICE_1, sync))
    p2 = mp.Process(target=_rank_main, args=(RANK_2, DEVICE_2, sync))
    p3 = mp.Process(target=_rank_main, args=(RANK_3, DEVICE_3, sync))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()

    if p0.exitcode != 0 or p1.exitcode != 0 or p2.exitcode != 0 or p3.exitcode != 0:
        raise RuntimeError(f"child rank failed: p0={p0.exitcode}, p1={p1.exitcode}, p2={p2.exitcode}, p3={p3.exitcode}")
    print("shared_dram_offload: all ranks OK", flush=True)


if __name__ == "__main__":
    main()
