add_executable(01_opencv_preprocess src/main.cpp)
target_link_libraries(01_opencv_preprocess PRIVATE ng_common ${OpenCV_LIBS})
target_include_directories(01_opencv_preprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
