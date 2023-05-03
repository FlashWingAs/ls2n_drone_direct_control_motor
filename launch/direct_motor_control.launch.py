from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from ament_index_python.packages import get_package_share_directory
import os

model = "tilthex"

def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "drone_namespace",
                default_value="Drone1",
                description="Drone namespace",
            ),
            Node(
                package="ls2n_drone_direct_motor_control",
                executable="direct_motor_control",
                output="screen",
                namespace=LaunchConfiguration("drone_namespace"),
                parameters = [RewrittenYaml(
                    source_file=os.path.join(
                    get_package_share_directory("ls2n_drone_sitl"),
                    "config",
                    "sitl_" + model + "_params.yaml",
                ),
                root_key="Drone1",
                param_rewrites={},
                )]
            )
        ]
    )
