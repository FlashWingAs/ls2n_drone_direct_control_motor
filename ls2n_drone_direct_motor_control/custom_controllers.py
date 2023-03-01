import numpy as np
import transforms3d as tf3d
from geometry_msgs.msg import Vector3Stamped
from ls2n_interfaces.msg import MotorControlSetPoint

from ls2n_drone_direct_motor_control.custom_common import Pose, Custom_Controller_Type

class Custom_Controller:

    def __init__(self, node):
        self.node = node
    
    type = Custom_Controller_Type.NONE
    desired_pose = Pose()

    def update_trajectory_setpoint(self):
        self.desired_pose.position = np.array([0.0, 0.0, 2.0]) #fixed set point of 2m altitude

    def do_control(self, _):
        pass

class Geometric_Controller(Custom_Controller):

    def __init__(self, node):
        super().__init__(node)
        self.type = Custom_Controller_Type.GEOMETRIC

    k_p_pos = 1
    k_i_pos = 0
    k_d_pos = 0
    k_p_ang = 1
    k_i_ang = 0
    k_d_ang = 0

    def do_control(self):
        desired_motor_speed = [0.5, 0.0001, 0.5, 0.0001, 0.0001, 0.0001]
        return desired_motor_speed