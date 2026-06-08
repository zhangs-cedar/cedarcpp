add_executable(07_plane_fit src/plane_fit.cpp)
target_link_libraries(07_plane_fit PRIVATE ng_common)

if(NG_BUILD_OPENCV)
  add_executable(07_stereo_depth src/stereo_depth.cpp)
  target_link_libraries(07_stereo_depth PRIVATE ng_common ${OpenCV_LIBS})
  target_include_directories(07_stereo_depth PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
