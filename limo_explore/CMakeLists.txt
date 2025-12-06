cmake_minimum_required(VERSION 3.0.2)
project(limo_explore)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  message_generation
  move_base_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

generate_messages(
  DEPENDENCIES std_msgs geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib geometry_msgs move_base_msgs nav_msgs roscpp rospy sensor_msgs std_msgs tf2_geometry_msgs tf2_ros visualization_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/cam_coverage.py
  scripts/frontier_planner.py
  scripts/straight_planner.py
  scripts/object_finder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
