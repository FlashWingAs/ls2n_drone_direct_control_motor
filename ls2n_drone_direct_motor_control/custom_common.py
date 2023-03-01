import numpy as np
from enum import IntEnum
from ls2n_interfaces.msg import DroneStatus
from ls2n_interfaces.srv import DroneRequest
from rclpy.qos import qos_profile_sensor_data

class Pose:
    time = 0.0
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    acceleration = np.array([0.0, 0.0, 0.0])

    orientation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

class Custom_Controller_Type(IntEnum):
    NONE = (0,)
    GEOMETRIC = 1