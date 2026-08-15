/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "acc_offload.h"

namespace py = pybind11;

void DefineAccOffloadConfig(py::module_ &m)
{
    py::enum_<offload_scene_t>(m, "Scene")
        .value("LOCAL", OFFLOAD_SCENE_LOCAL)
        .value("SHARED", OFFLOAD_SCENE_SHARED)
        .export_values();

    py::class_<offload_config_t>(m, "OffloadConfig")
        .def(py::init<>())
        .def_readwrite("device_id", &offload_config_t::deviceId)
        .def_readwrite("reserve_size", &offload_config_t::reserveSize,
                       "Reserved DRAM pool size in bytes, will be aligned up to GB")
        .def_readwrite("alloc_size", &offload_config_t::allocSize,
                       "Allocated local physical DRAM size in bytes, will be aligned up to GB. "
                       "LOCAL: must equal reserve_size; SHARED: provides the actual size")
        .def_readwrite("world_size", &offload_config_t::worldSize,
                       "number of ranks in the group (multi-card shared mode)")
        .def_readwrite("rank_id", &offload_config_t::rankId,
                       "local rank id, 0 is the server (multi-card shared mode)")
        .def_readwrite("scene", &offload_config_t::scene,
                       "memory pool scene: LOCAL=single-card, SHARED=multi-card shared")
        .def_readwrite("register_host_memory", &offload_config_t::registerHostMemory,
                       "register the local host pool and retain a device-visible address");
}

void DefineAccOffloadApi(py::module_ &m)
{
    m.def("initialize", &offload_init, py::call_guard<py::gil_scoped_release>(), py::arg("config"));

    m.def("uninitialize", &offload_uninit, py::call_guard<py::gil_scoped_release>());

    m.def("malloc", &offload_malloc, py::call_guard<py::gil_scoped_release>(), py::arg("size"), py::arg("flags") = 0);

    m.def("free", &offload_free, py::call_guard<py::gil_scoped_release>(), py::arg("ptr"), py::arg("flags") = 0);

    m.def("get_device_address", &offload_get_device_address, py::call_guard<py::gil_scoped_release>(),
          py::arg("ptr"), py::arg("size"));

    m.def("sparse_copy", &offload_sparse_copy, py::call_guard<py::gil_scoped_release>(),
          py::arg("srcPtrs"), py::arg("dstPtrs"), py::arg("lenPtrs"), py::arg("sizePtr"), py::arg("deviceId"));

    m.def("lru_resident_compact", &offload_lru_resident_compact, py::call_guard<py::gil_scoped_release>(),
          py::arg("req_ids"), py::arg("last_req_ids"), py::arg("topk_indices"), py::arg("stable_prefix_lens"),
          py::arg("slot_to_token"), py::arg("lru_slots"), py::arg("current_slots"), py::arg("miss_count"),
          py::arg("miss_tokens"), py::arg("miss_slots"), py::arg("token_mark_workspace"),
          py::arg("token_pos_workspace"), py::arg("epochs"), py::arg("num_reqs"), py::arg("topk"),
          py::arg("capacity"), py::arg("max_token"), py::arg("deviceId"));

    m.def("compute_lru_resident_addrs", &offload_compute_lru_resident_addrs,
          py::call_guard<py::gil_scoped_release>(), py::arg("miss_count"), py::arg("miss_tokens"),
          py::arg("miss_slots"), py::arg("block_table"), py::arg("gvas_buffer"), py::arg("addr_buffer"),
          py::arg("size_buffer"), py::arg("num_tokens_buffer"), py::arg("block_size"),
          py::arg("token_size_bytes_k"), py::arg("token_size_bytes_v"), py::arg("gvas_k_base"),
          py::arg("gvas_v_base"), py::arg("addr_k_base"), py::arg("addr_v_base"), py::arg("resident_capacity"),
          py::arg("num_reqs"), py::arg("topk"), py::arg("max_num_blocks"), py::arg("deviceId"));

    m.def("get_sparse_kv_plan_workspace_size",
          &offload_get_sparse_kv_plan_workspace_size,
          py::arg("num_reqs"), py::arg("topk"), py::arg("capacity"));

    m.def("get_sparse_kv_fsa_row_map_workspace_size",
          &offload_get_sparse_kv_fsa_row_map_workspace_size,
          py::arg("num_logical_rows"), py::arg("physical_row_capacity"));
    m.def("get_sparse_kv_fsa_plan_row_stride",
          &offload_get_sparse_kv_fsa_plan_row_stride, py::arg("topk"));

    m.def("sparse_kv_load_runtime",
          [](uint64_t req_ids, uint64_t last_req_ids,
             uint64_t topk_indices, uint64_t stable_prefix_lens,
             uint64_t slot_to_token, uint64_t lru_slots,
             uint64_t current_slots, uint64_t miss_count,
             uint64_t miss_tokens, uint64_t miss_slots,
             uint64_t compact_workspace,
             uint64_t compact_workspace_bytes, uint64_t block_table,
             uint64_t host_k_base, uint64_t host_v_base,
             uint64_t device_k_base, uint64_t device_v_base,
             int64_t num_reqs, int64_t topk, int64_t capacity,
             int64_t max_token, int64_t max_num_blocks,
             int32_t block_size, int32_t token_size_bytes_k,
             int32_t token_size_bytes_v, uint16_t deviceId) {
              sparse_kv_load_runtime_params_t params{};
              params.req_ids = req_ids;
              params.last_req_ids = last_req_ids;
              params.topk_indices = topk_indices;
              params.stable_prefix_lens = stable_prefix_lens;
              params.slot_to_token = slot_to_token;
              params.lru_slots = lru_slots;
              params.current_slots = current_slots;
              params.miss_count = miss_count;
              params.miss_tokens = miss_tokens;
              params.miss_slots = miss_slots;
              params.compact_workspace = compact_workspace;
              params.compact_workspace_bytes = compact_workspace_bytes;
              params.block_table = block_table;
              params.host_k_base = host_k_base;
              params.host_v_base = host_v_base;
              params.device_k_base = device_k_base;
              params.device_v_base = device_v_base;
              params.num_reqs = num_reqs;
              params.topk = topk;
              params.capacity = capacity;
              params.max_token = max_token;
              params.max_num_blocks = max_num_blocks;
              params.block_size = block_size;
              params.token_size_bytes_k = token_size_bytes_k;
              params.token_size_bytes_v = token_size_bytes_v;
              return offload_sparse_kv_load_runtime(&params, deviceId);
          }, py::call_guard<py::gil_scoped_release>(),
          py::arg("req_ids"), py::arg("last_req_ids"),
          py::arg("topk_indices"), py::arg("stable_prefix_lens"),
          py::arg("slot_to_token"), py::arg("lru_slots"),
          py::arg("current_slots"), py::arg("miss_count"),
          py::arg("miss_tokens"), py::arg("miss_slots"),
          py::arg("compact_workspace"),
          py::arg("compact_workspace_bytes"), py::arg("block_table"),
          py::arg("host_k_base"), py::arg("host_v_base"),
          py::arg("device_k_base"), py::arg("device_v_base"),
          py::arg("num_reqs"), py::arg("topk"), py::arg("capacity"),
          py::arg("max_token"), py::arg("max_num_blocks"),
          py::arg("block_size"), py::arg("token_size_bytes_k"),
          py::arg("token_size_bytes_v"), py::arg("deviceId"));

    m.def("sparse_kv_plan_fsa_runtime",
          [](uint64_t req_ids, uint64_t last_req_ids,
             uint64_t topk_indices, uint64_t stable_prefix_lens,
             uint64_t visible_seq_lens, uint64_t slot_to_token,
             uint64_t lru_slots, uint64_t current_slots,
             uint64_t miss_count, uint64_t miss_tokens,
             uint64_t miss_slots, uint64_t compact_workspace,
             uint64_t compact_workspace_bytes,
             uint64_t row_map_workspace,
             uint64_t row_map_workspace_bytes, uint64_t encoded_plan,
             uint64_t current_linear_slots, int64_t num_logical_rows,
             int64_t physical_row_capacity, int64_t topk,
             int64_t capacity, int64_t max_token,
             int64_t encoded_plan_stride, uint16_t deviceId) {
              sparse_kv_plan_fsa_runtime_params_t params{};
              params.req_ids = req_ids;
              params.last_req_ids = last_req_ids;
              params.topk_indices = topk_indices;
              params.stable_prefix_lens = stable_prefix_lens;
              params.visible_seq_lens = visible_seq_lens;
              params.slot_to_token = slot_to_token;
              params.lru_slots = lru_slots;
              params.current_slots = current_slots;
              params.miss_count = miss_count;
              params.miss_tokens = miss_tokens;
              params.miss_slots = miss_slots;
              params.compact_workspace = compact_workspace;
              params.compact_workspace_bytes = compact_workspace_bytes;
              params.row_map_workspace = row_map_workspace;
              params.row_map_workspace_bytes = row_map_workspace_bytes;
              params.encoded_plan = encoded_plan;
              params.current_linear_slots = current_linear_slots;
              params.num_logical_rows = num_logical_rows;
              params.physical_row_capacity = physical_row_capacity;
              params.topk = topk;
              params.capacity = capacity;
              params.max_token = max_token;
              params.encoded_plan_stride = encoded_plan_stride;
              return offload_sparse_kv_plan_fsa_runtime(&params, deviceId);
          }, py::call_guard<py::gil_scoped_release>(),
          py::arg("req_ids"), py::arg("last_req_ids"),
          py::arg("topk_indices"), py::arg("stable_prefix_lens"),
          py::arg("visible_seq_lens"), py::arg("slot_to_token"),
          py::arg("lru_slots"), py::arg("current_slots"),
          py::arg("miss_count"), py::arg("miss_tokens"),
          py::arg("miss_slots"), py::arg("compact_workspace"),
          py::arg("compact_workspace_bytes"),
          py::arg("row_map_workspace"),
          py::arg("row_map_workspace_bytes"), py::arg("encoded_plan"),
          py::arg("current_linear_slots"), py::arg("num_logical_rows"),
          py::arg("physical_row_capacity"), py::arg("topk"),
          py::arg("capacity"), py::arg("max_token"),
          py::arg("encoded_plan_stride"), py::arg("deviceId"));
}

PYBIND11_MODULE(_pymf_acc_offload, m)
{
    auto offload = m.def_submodule("offload", "Acc Offload Module.");

    DefineAccOffloadConfig(offload);
    DefineAccOffloadApi(offload);
}

#pragma GCC diagnostic pop
