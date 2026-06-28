cmake_minimum_required(VERSION 3.8)
project(grasp_vision_pkg)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(robot_interface_pkg REQUIRED)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)

install(DIRECTORY ${PROJECT_NAME}
  DESTINATION ${PYTHON_INSTALL_DIR}
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/camera_subscriber
  scripts/grasp_pose_estimator
  scripts/sam3_onnx_segmenter
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
