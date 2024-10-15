cmake_minimum_required(VERSION 3.5)
project(simple_loop_closure)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)

include_directories(
  include
  thirdparty/nanoflann/include
  ${GTSAM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME}_node src/simple_loop_closure_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_node
  "pcl_conversions"
  "rcl_interfaces"
  "rclcpp"
  "rcutils"
  "sensor_msgs"
  "nav_msgs"
  "std_msgs"
  "geometry_msgs"
  "pcl_ros"
  "visualization_msgs"
  "tf2"
  "tf2_eigen"
)
target_link_libraries(${PROJECT_NAME}_node
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  gtsam
)
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# add_executable(test_lc_node src/test.cpp)
# ament_target_dependencies(test_lc_node
#   "pcl_conversions"
#   "rcl_interfaces"
#   "rclcpp"
#   "rcutils"
#   "sensor_msgs"
#   "nav_msgs"
#   "std_msgs"
#   "geometry_msgs"
#   "pcl_ros"
#   "visualization_msgs"
#   "tf2"
#   "tf2_eigen"
# )
# target_link_libraries(test_lc_node
#   ${Boost_LIBRARIES}
#   ${PCL_LIBRARIES}
#   gtsam
# )
# install(TARGETS
#   test_lc_node
#   DESTINATION lib/${PROJECT_NAME}
# )

install(DIRECTORY config launch 
  DESTINATION share/${PROJECT_NAME} 
)

ament_package()