import numpy as np
from enum import IntEnum
from ls2n_interfaces.msg import DroneStatus
from ls2n_interfaces.srv import DroneRequest
from rclpy.qos import qos_profile_sensor_data

class Custom_Pose:
    time = 0.0
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    acceleration = np.array([0.0, 0.0, 0.0])

    rotation = np.array([0.0, 0.0, 0.0, 0.0])
    rot_velocity = np.array([0.0, 0.0, 0.0])
    rot_acceleration = np.array([0.0, 0.0, 0.0])
    rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

class Custom_Controller_Type(IntEnum):
    NONE = (0,)
    TEST = (1,)
    GEOMETRIC = 2

def wedge_op(A):
    a1 = A[2, 1]
    a2 = A[0, 2]
    a3 = A[1, 0]
    a_x = np.array([a1, a2, a3])
    return(a_x)