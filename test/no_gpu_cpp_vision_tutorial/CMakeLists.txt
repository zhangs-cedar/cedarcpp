cmake_minimum_required(VERSION 3.16)
project(no_gpu_cpp_vision_tutorial LANGUAGES CXX)

option(NG_BUILD_OPENCV "Build OpenCV examples" ON)
option(NG_BUILD_ORT "Build ONNXRuntime CPU example" OFF)
option(NG_BUILD_PYBIND11 "Build pybind11 example" OFF)
option(NG_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(ng_common INTERFACE)
target_include_directories(ng_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

if(MSVC)
  target_compile_options(ng_common INTERFACE /W4)
  if(NG_WARNINGS_AS_ERRORS)
    target_compile_options(ng_common INTERFACE /WX)
  endif()
else()
  target_compile_options(ng_common INTERFACE -Wall -Wextra -Wpedantic)
  if(NG_WARNINGS_AS_ERRORS)
    target_compile_options(ng_common INTERFACE -Werror)
  endif()
endif()

if(NG_BUILD_OPENCV OR NG_BUILD_ORT)
  find_package(OpenCV REQUIRED)
  message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

add_subdirectory(00_linux_cpp_cmake)

if(NG_BUILD_OPENCV)
  add_subdirectory(01_opencv_preprocess)
  add_subdirectory(02_roi_measure)
endif()

if(NG_BUILD_ORT)
  add_subdirectory(03_onnxruntime_cpu_infer)
endif()

add_subdirectory(04_infer_interface_stub)
add_subdirectory(05_multithread_pipeline)

if(NG_BUILD_PYBIND11)
  add_subdirectory(06_pybind11_infer)
endif()

add_subdirectory(07_3d_measurement)
