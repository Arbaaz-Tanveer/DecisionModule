cmake_minimum_required(VERSION 3.5)
project(o5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(ompl REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(ACADO REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${ACADO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Add executables
add_executable(main5 src/main5.cpp src/Visualize.cpp)
add_executable(controller5 src/controller5.cpp)

# Configure main executable
ament_target_dependencies(main5
  rclcpp
  geometry_msgs
  std_msgs
)

target_link_libraries(main5
  ${OMPL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  Boost::program_options
  Boost::system
  Boost::filesystem
)

# Configure controller executable
ament_target_dependencies(controller5
  rclcpp
  std_msgs
  geometry_msgs
  pluginlib
)

target_link_libraries(controller5
  ${OMPL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${ACADO_SHARED_LIBRARIES}
  Eigen3::Eigen
  Boost::program_options
  Boost::system
  Boost::filesystem
)

target_include_directories(controller5 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_compile_features(controller5 PUBLIC c_std_99 cxx_std_17)

# Install targets
install(TARGETS
  main5
  controller5
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
