#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_sample_data.py
cmake -S . -B build -G Ninja -DNG_BUILD_OPENCV=ON -DNG_BUILD_ORT=OFF -DNG_BUILD_PYBIND11=OFF
cmake --build build -j

./build/00_linux_cpp_cmake/00_scan_files --input data/images
./build/01_opencv_preprocess/01_opencv_preprocess --input data/images --output out/01_preprocess
./build/02_roi_measure/02_roi_measure --input data/images --output out/02_roi
./build/04_infer_interface_stub/04_infer_interface_stub --backend cpu
./build/05_multithread_pipeline/05_multithread_pipeline --input data/images --output out/05_pipeline --repeat 1000
./build/07_3d_measurement/07_plane_fit --input data/pointcloud/plane.xyz --output out/07_plane.json
./build/07_3d_measurement/07_stereo_depth --left data/stereo/left.png --right data/stereo/right.png --output out/07_stereo

echo "basic acceptance passed"
