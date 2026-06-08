#include <numeric>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ng/common/timer.hpp"

namespace py = pybind11;

struct PyInferResult {
    int class_id = 0;
    float score = 0.0f;
    double cost_ms = 0.0;
    std::string backend = "cpp_cpu_fake";
};

class Engine {
public:
    explicit Engine(std::string model_path) : model_path_(std::move(model_path)) {}

    PyInferResult infer_vector(const std::vector<float>& input) const {
        ng::Timer timer;
        const float sum = std::accumulate(input.begin(), input.end(), 0.0f);
        const float mean = input.empty() ? 0.0f : sum / static_cast<float>(input.size());
        PyInferResult r;
        r.class_id = mean > 0.5f ? 1 : 0;
        r.score = mean;
        r.cost_ms = timer.elapsed_ms();
        return r;
    }

    std::string model_path() const { return model_path_; }

private:
    std::string model_path_;
};

PYBIND11_MODULE(ng_py_infer, m) {
    m.doc() = "No-GPU C++ inference module exposed by pybind11";

    py::class_<PyInferResult>(m, "InferResult")
        .def_readonly("class_id", &PyInferResult::class_id)
        .def_readonly("score", &PyInferResult::score)
        .def_readonly("cost_ms", &PyInferResult::cost_ms)
        .def_readonly("backend", &PyInferResult::backend)
        .def("__repr__", [](const PyInferResult& r) {
            return "InferResult(class_id=" + std::to_string(r.class_id) +
                   ", score=" + std::to_string(r.score) +
                   ", cost_ms=" + std::to_string(r.cost_ms) +
                   ", backend='" + r.backend + "')";
        });

    py::class_<Engine>(m, "Engine")
        .def(py::init<std::string>())
        .def("infer_vector", &Engine::infer_vector)
        .def_property_readonly("model_path", &Engine::model_path);
}
