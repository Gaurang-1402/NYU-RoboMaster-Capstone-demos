cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0057 NEW)
project(demo_cv)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_PREFIX_PATH "~/libtorch/share/cmake/Torch")

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

target_link_libraries(traditional_cv "${TORCH_LIBRARIES}")
target_link_libraries(traditional_cv TorchVision::TorchVision)


install(TARGETS
  image_publisher
  traditional_cv
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
