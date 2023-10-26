cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0057 NEW)
project(demo_cv)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

enable_language(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

list(APPEND CMAKE_PREFIX_PATH "~/Downloads/libtorch")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(TorchVision REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Add your nodes
add_executable(image_publisher src/image_publisher.cpp)
ament_target_dependencies(image_publisher rclcpp sensor_msgs cv_bridge OpenCV ament_index_cpp)

add_executable(traditional_cv src/traditional_cv.cpp)
ament_target_dependencies(traditional_cv rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(video_publisher src/video_publisher.cpp)
ament_target_dependencies(video_publisher rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(deeplearning_cv src/deeplearning_cv.cpp)
ament_target_dependencies(deeplearning_cv rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(comp_traditional_cv src/comp_traditional_cv.cpp)
ament_target_dependencies(comp_traditional_cv rclcpp sensor_msgs cv_bridge OpenCV)

target_link_libraries(traditional_cv "${TORCH_LIBRARIES}")
target_link_libraries(traditional_cv TorchVision::TorchVision)

target_link_libraries(deeplearning_cv "${TORCH_LIBRARIES}")
target_link_libraries(deeplearning_cv TorchVision::TorchVision)

target_link_libraries(comp_traditional_cv "${TORCH_LIBRARIES}")
target_link_libraries(comp_traditional_cv TorchVision::TorchVision)

target_link_libraries(video_publisher "${TORCH_LIBRARIES}")
target_link_libraries(video_publisher TorchVision::TorchVision)


install(TARGETS
  image_publisher
  traditional_cv
  deeplearning_cv
  video_publisher
  comp_traditional_cv
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
