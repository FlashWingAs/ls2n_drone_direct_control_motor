import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from ls2n_drone_direct_motor_control.custom_common import Custom_Pose

class Custom_Observer:
    
    def __init__(self, node, m_tot, I) -> None:
        self.node = node
        self.g = np.array([0, 0, m_tot*9.81])
        self.M = np.concatenate((np.concatenate((np.diag(m_tot*np.ones(3)), np.zeros((3, 3))), 0), np.concatenate((np.zeros((3, 3)), I), 0)), 1)
        self.K0 = 2.0
        self.t = 0.0
        self.t_old = 0.0
        self.delta_t = 0.0
        self.actuator_wrench_int = np.zeros(6)
        self.estimated_ext_wrench_int = np.zeros(6)
        self.estimated_wrench_old = np.zeros(6)
        self.estimated_wrench = np.zeros(6)
        self.gravity_int = np.zeros(6)

    def do_observer(self, real_pose : Custom_Pose, desired_wrench, timestamp):
        self.estimated_wrench_old = self.estimated_wrench
        self.t_old = self.t
        self.t = timestamp
        self.delta_t = self.t - self.delta_t
        rotation_matrix = R.from_quat(np.roll(real_pose.rotation.elements, -1)).as_matrix()
        v = np.concatenate((np.linalg.inv(rotation_matrix) @ real_pose.velocity, real_pose.rot_velocity))
        self.actuator_wrench_int += desired_wrench*self.delta_t
        self.estimated_ext_wrench_int += self.estimated_wrench_old*self.delta_t
        self.gravity_int += np.concatenate((np.linalg.inv(rotation_matrix) @ self.g, np.zeros(3)))*self.delta_t
        self.estimated_wrench = self.K0*(self.M @ v - (self.actuator_wrench_int + self.estimated_ext_wrench_int - self.gravity_int))
        # self.node.get_logger().info("delta_t = "+str(self.delta_t))
        # self.node.get_logger().info("rotation_matrix = "+str(rotation_matrix))
        # self.node.get_logger().info("estimated wrench = "+str(self.estimated_wrench))