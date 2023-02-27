from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="ls2n_drone_direct_motor_control",
                executable="direct_motor_control"
            )
        ]
    )
