import numpy as np
from enum import IntEnum
from ls2n_interfaces.msg import DroneStatus
from ls2n_interfaces.srv import DroneRequest
from rclpy.qos import qos_profile_sensor_data
from pyquaternion import Quaternion
import copy

class Custom_Pose:
    def __init__(self) -> None:
        self.time = 0.0                                                                 # pose timestamp
        self.position = np.array([0.0, 0.0, 0.0])                                            # 3D-vector of position
        self.velocity = np.array([0.0, 0.0, 0.0])                                            # 3D-vector of velocity
        self.acceleration = np.array([0.0, 0.0, 0.0])                                        # 3D-vector of acceleration

        self.rotation = Quaternion()                                                         # quaternion of orientation
        self.rot_velocity = np.array([0.0, 0.0, 0.0])                                        # 3D-vector of instantaneous rotation speed
        self.rot_acceleration = np.array([0.0, 0.0, 0.0])                                    # 3D-vector of instantaneous rotation acceleration

    def copy(self):
        return copy.deepcopy(self)

class Custom_Controller_Type(IntEnum):
    NONE = (0,)
    TEST = (1,)
    GEOMETRIC = (2,)
    VELOCITY = (3)

class Custom_PID_Param:
    def __init__(self,
                 trans_p = np.array([0.0, 0.0, 0.0]), trans_i = np.array([0.0, 0.0, 0.0]), trans_d = np.array([0.0, 0.0, 0.0]),
                 rot_p = np.array([0.0, 0.0, 0.0]), rot_i = np.array([0.0, 0.0, 0.0]), rot_d = np.array([0.0, 0.0, 0.0])) -> None:
        self.trans_p = trans_p
        self.trans_i = trans_i
        self.trans_d = trans_d
        self.rot_p = rot_p
        self.rot_i = rot_i
        self.rot_d = rot_d
        self.Ktp = np.diag(self.trans_p)
        self.Kti = np.diag(self.trans_i)
        self.Ktd = np.diag(self.trans_d)
        self.Krp = np.diag(self.rot_p)
        self.Kri = np.diag(self.rot_i)
        self.Krd = np.diag(self.rot_d)
    
    def update(self):
        self.Ktp = np.diag(self.trans_p)
        self.Kti = np.diag(self.trans_i)
        self.Ktd = np.diag(self.trans_d)
        self.Krp = np.diag(self.rot_p)
        self.Kri = np.diag(self.rot_i)
        self.Krd = np.diag(self.rot_d)