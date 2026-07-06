from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('grasp_vision_pkg'),
                'config',
                'config.yaml',
            ]),
            description='Path to the camera subscriber parameter file.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('preload_grasp_estimator', default_value='true'),
        DeclareLaunchArgument('background_object_bbox_filter', default_value='true'),
        DeclareLaunchArgument('background_object_bbox_padding_xy', default_value='0.06'),
        DeclareLaunchArgument('background_object_bbox_padding_z', default_value='0.04'),
        Node(
            package='grasp_vision_pkg',
            executable='camera_subscriber',
            name='camera_subscriber',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                    'preload_grasp_estimator': ParameterValue(
                        LaunchConfiguration('preload_grasp_estimator'),
                        value_type=bool,
                    ),
                    'background_object_bbox_filter': ParameterValue(
                        LaunchConfiguration('background_object_bbox_filter'),
                        value_type=bool,
                    ),
                    'background_object_bbox_padding_xy': ParameterValue(
                        LaunchConfiguration('background_object_bbox_padding_xy'),
                        value_type=float,
                    ),
                    'background_object_bbox_padding_z': ParameterValue(
                        LaunchConfiguration('background_object_bbox_padding_z'),
                        value_type=float,
                    ),
                },
            ],
        ),
    ])
