cmake_minimum_required(VERSION 3.10)

project(gpu_test)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} opencv_gpu_test.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(gpu_vs_cpu gpu_vs_cpu_live_video.cpp)
target_link_libraries(gpu_vs_cpu ${OpenCV_LIBS})

add_executable(basic_operation_gpu basic_gpu_operation_opencv.cpp)
target_link_libraries(basic_operation_gpu ${OpenCV_LIBS})

add_executable(mono_depth_map monocular_camera_depth_map.cpp)
target_link_libraries(mono_depth_map ${OpenCV_LIBS})