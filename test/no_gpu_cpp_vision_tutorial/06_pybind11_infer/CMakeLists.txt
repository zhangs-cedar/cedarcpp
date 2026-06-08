find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ng_py_infer src/bindings.cpp)
target_link_libraries(ng_py_infer PRIVATE ng_common)
