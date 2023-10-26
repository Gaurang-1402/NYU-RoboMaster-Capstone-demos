cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0057 NEW)
project(camera_architecture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_PREFIX_PATH "~/Downloads/libtorch")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Add your nodes
add_executable(camera_publisher src/camera_publisher.cpp)
ament_target_dependencies(camera_publisher rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(image_display src/image_display.cpp)
ament_target_dependencies(image_display rclcpp sensor_msgs cv_bridge OpenCV)

# target_link_libraries(faster_rcnn "${TORCH_LIBRARIES}")
target_link_libraries(image_display "${TORCH_LIBRARIES}")

install(TARGETS
  camera_publisher
  image_display
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
