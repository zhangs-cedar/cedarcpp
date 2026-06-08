add_executable(02_roi_measure src/main.cpp)
target_link_libraries(02_roi_measure PRIVATE ng_common ${OpenCV_LIBS})
target_include_directories(02_roi_measure PRIVATE ${OpenCV_INCLUDE_DIRS})
