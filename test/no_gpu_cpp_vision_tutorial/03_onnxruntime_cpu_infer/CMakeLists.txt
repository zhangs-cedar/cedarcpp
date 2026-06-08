if(NOT DEFINED ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "NG_BUILD_ORT=ON requires -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include
  NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib
  NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "Cannot find ONNXRuntime under ${ONNXRUNTIME_ROOT}")
endif()

add_executable(03_ort_cpu_infer src/main.cpp src/ort_engine.cpp)
target_include_directories(03_ort_cpu_infer PRIVATE include ${ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(03_ort_cpu_infer PRIVATE ng_common ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
