import numpy as np
from enum import IntEnum
from ls2n_interfaces.msg import DroneStatus
from ls2n_interfaces.srv import DroneRequest
from rclpy.qos import qos_profile_sensor_data
from pyquaternion import Quaternion
import copy
from scipy.spatial.transform import Rotation as R

class Tilthex_Pose:
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

class Tilthex_Controller_Type(IntEnum):
    NONE = (0,)
    TEST = (1,)
    GEOMETRIC = (2,)
    VELOCITY = (3)

class Tilthex_PID_Param:
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

class Tilthex_Observer:
    
    def __init__(self, node, m_tot, I) -> None:
        self.node = node
        self.g = np.array([0, 0, m_tot*9.81])
        self.M = np.concatenate((np.concatenate((np.diag(m_tot*np.ones(3)), np.zeros((3, 3))), 0), np.concatenate((np.zeros((3, 3)), I), 0)), 1)
        self.I = I
        self.K0 = 3.0
        self.t = 0.0
        self.t_old = 0.0
        self.delta_t = 0.0
        self.integral = np.zeros(6)
        self.correct_desired_wrench = np.zeros(6)
        self.estimated_wrench_old = np.zeros(6)
        self.estimated_wrench = np.zeros(6)
        self.c = np.zeros(6)

    def do_observer(self, real_pose : Tilthex_Pose, desired_wrench, timestamp):
        # Given desired wrench is in drone reference. We need to move the force to the world reference
        rotation_matrix = R.from_quat(np.roll(real_pose.rotation.elements, -1)).as_matrix()
        self.correct_desired_wrench = np.concatenate((rotation_matrix@desired_wrench[0:3], desired_wrench[3:6]))
        self.estimated_wrench_old = np.copy(self.estimated_wrench)
        self.t_old = self.t
        self.t = timestamp
        self.delta_t = self.t - self.t_old
        v = np.concatenate((real_pose.velocity, real_pose.rot_velocity))
        self.c = np.concatenate((self.g, np.cross(real_pose.rot_velocity, self.I @ real_pose.rot_velocity)))
        self.integral += (self.c - self.correct_desired_wrench - self.estimated_wrench_old)*self.delta_t
        self.estimated_wrench = self.K0 * self.M @ v + self.K0 * self.integral
        # self.node.get_logger().info("delta_t = "+str(self.delta_t))
        # self.node.get_logger().info("actuator_wrench = "+str(desired_wrench))
        # self.node.get_logger().info("c = "+str(self.c))
        # self.node.get_logger().info("integral = "+str(self.integral))
        # self.node.get_logger().info("estimated wrench = "+str(self.estimated_wrench))