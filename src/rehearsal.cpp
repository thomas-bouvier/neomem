#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "engine_loader.hpp"
#include "distributed_stream_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neomem, m) {
    m.doc() = "Neomem: pybind11 based streaming rehearsal buffer";

    py::class_<engine_loader_t>(m, "EngineLoader")
        .def(py::init<std::string, uint16_t, bool>(), py::call_guard<py::scoped_ostream_redirect>())
        .def("wait_for_finalize", &engine_loader_t::wait_for_finalize, py::call_guard<py::gil_scoped_release>());

    py::class_<distributed_stream_loader_t>(m, "DistributedStreamLoader")
        // Return policy is set to `reference` instead of `take_ownership`
        // because finalization callbacks are ensuring that providers are freed.
        .def_static("create", &distributed_stream_loader_t::create, py::return_value_policy::reference)
        .def("register_endpoints", &distributed_stream_loader_t::register_endpoints)
        .def("use_these_allocated_variables", py::overload_cast<const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &>(&distributed_stream_loader_t::use_these_allocated_variables))
        .def("use_these_allocated_variables", py::overload_cast<const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &, const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::use_these_allocated_variables))
        .def("enable_augmentation", &distributed_stream_loader_t::enable_augmentation)
        .def("measure_performance", &distributed_stream_loader_t::measure_performance)
        .def("start", &distributed_stream_loader_t::start)
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &, const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("wait", &distributed_stream_loader_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("get_rehearsal_size", &distributed_stream_loader_t::get_rehearsal_size, py::call_guard<py::gil_scoped_release>())
        .def("get_metrics", &distributed_stream_loader_t::get_metrics, py::call_guard<py::gil_scoped_release>())
        .def("finalize", &distributed_stream_loader_t::finalize, py::call_guard<py::gil_scoped_release>());

    py::enum_<Task>(m, "Task")
        .value("Classification", Classification)
        .value("Reconstruction", Reconstruction)
        .export_values();
    
    py::enum_<BufferStrategy>(m, "BufferStrategy")
        .value("NoBuffer", NoBuffer)
        .value("CPUBuffer", CPUBuffer)
        .value("CUDABuffer", CUDABuffer)
        .export_values();
}
