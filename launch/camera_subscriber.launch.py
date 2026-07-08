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
                },
            ],
        ),
    ])
