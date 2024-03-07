cmake_minimum_required(VERSION 3.0.2)
project(samcon_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)


add_message_files(
  FILES
  Glass_Gaze.msg
  Glass_Gait.msg
)



generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES samcon_perception
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  scripts/glass_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
# catkin_install_python(PROGRAMS
#   scripts/stereo_camera_sub.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
catkin_install_python(PROGRAMS
  scripts/imu/imu_.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
catkin_install_python(PROGRAMS
  scripts/imu/pose_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
catkin_install_python(PROGRAMS
  scripts/glass_depth_bridge2.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
