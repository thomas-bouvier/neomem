#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "distributed_stream_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neomem, m) {
    m.doc() = "Neomem: pybind11 based streaming rehearsal buffer";

    py::class_<engine_loader_t>(m, "EngineLoader")
        .def(py::init<std::string, uint16_t, bool>(), py::call_guard<py::scoped_ostream_redirect>());

    py::class_<distributed_stream_loader_t>(m, "DistributedStreamLoader")
        .def(py::init<engine_loader_t, Task, unsigned int, unsigned int, unsigned int, int64_t, unsigned int, std::vector<long>, bool, bool>(), py::call_guard<py::scoped_ostream_redirect>())
        .def("register_endpoints", &distributed_stream_loader_t::register_endpoints)
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("accumulate", py::overload_cast<const torch::Tensor &, const torch::Tensor &,
                 const torch::Tensor &, const torch::Tensor &, const torch::Tensor &>(&distributed_stream_loader_t::accumulate), py::call_guard<py::gil_scoped_release>())
        .def("wait", &distributed_stream_loader_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("use_these_allocated_variables", &distributed_stream_loader_t::use_these_allocated_variables, py::call_guard<py::gil_scoped_release>())
        .def("enable_augmentation", &distributed_stream_loader_t::enable_augmentation, py::call_guard<py::gil_scoped_release>())
        .def("get_rehearsal_size", &distributed_stream_loader_t::get_rehearsal_size, py::call_guard<py::gil_scoped_release>())
        .def("get_history_count", &distributed_stream_loader_t::get_history_count, py::call_guard<py::gil_scoped_release>())
        .def("get_metrics", &distributed_stream_loader_t::get_metrics, py::call_guard<py::gil_scoped_release>());

    py::enum_<Task>(m, "Task")
        .value("Classification", Classification)
        .value("Reconstruction", Reconstruction)
        .export_values();
}
