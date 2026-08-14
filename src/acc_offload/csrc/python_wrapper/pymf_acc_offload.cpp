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
}

PYBIND11_MODULE(_pymf_acc_offload, m)
{
    auto offload = m.def_submodule("offload", "Acc Offload Module.");

    DefineAccOffloadConfig(offload);
    DefineAccOffloadApi(offload);
}

#pragma GCC diagnostic pop
