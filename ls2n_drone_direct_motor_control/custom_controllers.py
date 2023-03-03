import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d as tf3d
from geometry_msgs.msg import Vector3Stamped
from ls2n_interfaces.msg import MotorControlSetPoint

from ls2n_drone_direct_motor_control.custom_common import Custom_Pose, Custom_Controller_Type

class Custom_Controller:

    def __init__(self, node):
        self.node = node
    
    type = Custom_Controller_Type.NONE
    desired_pose = Custom_Pose()

    def update_trajectory_setpoint(self):
        self.desired_pose.position = np.array([0.0, 0.0, 2.0]) #fixed set point of 2m altitude

    def do_control(self, _):
        pass

class Test_Controller(Custom_Controller):
    def __init__(self, node):
        super().__init__(node)
        self.type = Custom_Controller_Type.TEST

    def do_control(self):
        desired_motor_speed = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        return desired_motor_speed

class Geometric_Controller(Custom_Controller):

    def __init__(self, node):
        super().__init__(node)
        self.type = Custom_Controller_Type.GEOMETRIC
    
    Lx = 0.17
    alpha = np.pi/3
    
    k_p_pos = 1
    k_i_pos = 0
    k_d_pos = 0
    k_p_ang = 1
    k_i_ang = 0
    k_d_ang = 0

    e_pos_old = np.zeros((3,1))
    e_pos = np.zeros((3,1))
    D_e_pos = np.zeros((3,1))
    I_e_pos = np.zeros((3,1))

    e_ang_old = np.zeros((3,1))
    e_ang = np.zeros((3,1))
    D_e_ang = np.zeros((3,1))
    I_e_ang = np.zeros((3,1))
    g = 9.81
    m_tot = 1
    I = np.diag([1, 1, 1])
    I_inv = np.linalg.pinv(I)

    B_P_Pi = np.zeros((6,3,1))
    B_R_Pi = np.zeros((6,3,3))
    F = np.zeros((3,6))
    H = np.zeros((3,6))

    def do_control(self, real_pose : Custom_Pose, desired_pose : Custom_Pose, step_size):
        # Init
        
        pi = np.pi
        
        lambda_arms = np.array([-pi/2, pi/2, pi/6, -5*pi/6, -pi/6, 5*pi/6])
        alpha_arms = self.alpha*np.array([-1, 1, -1, 1, 1, -1])
        clock = np.array([1, -1, 1, -1, -1, 1])
        kf = 1
        kd = 1

        # updates

        self.D_e_pos = (desired_pose.position - self.e_pos)/step_size
        self.I_e_pos += desired_pose.position*step_size
        self.e_pos_old = self.e_pos
        self.e_pos = real_pose.position - desired_pose.position

        self.D_e_ang = (desired_pose.rotation - self.e_ang)/step_size
        self.I_e_ang += desired_pose.rotation*step_size
        self.e_ang_old = self.e_ang
        self.e_ang = real_pose.rotation - desired_pose.rotation


        # Calc V

        Kp1 = self.k_d_pos*np.eye(3)
        Kp2 = self.k_p_pos*np.eye(3)
        Kp3 = self.k_i_pos*np.eye(3)
        Kr1 = self.k_d_ang*np.eye(3)
        Kr2 = self.k_p_ang*np.eye(3)
        Kr3 = self.k_i_ang*np.eye(3)
        DD_p_d = desired_pose.acceleration
        DD_r_d = desired_pose.rot_acceleration

        Vp = DD_p_d - Kp1*self.D_e_pos - Kp2*self.e_pos - Kp3*self.I_e_pos
        Vr = DD_r_d - Kr1*self.D_e_ang - Kr2*self.e_ang - Kr3*self.I_e_ang
        v = np.concatenate((Vp, Vr), axis=0)

        # Calc f

        f = np.array([[0], [0], [-self.g*self.m_tot], [0], [0], [0]])

        # Calc J

        for i in range(6):
            self.B_P_Pi[i, :, :] = R.from_euler('ZYX', np.array([lambda_arms[i], 0, 0])).as_matrix * np.transpose([np.array([self.Lx, 0, 0])])
            self.B_R_Pi[i, :, :] = R.from_euler('ZYX', np.array([lambda_arms[i], 0, 0])).as_matrix * R.from_euler('ZYX', np.array([0, 0, alpha_arms[i]])).as_matrix
            self.F[:, i] = np.reshape(self.B_R_Pi[i, :, :], (3, 3)) * np.transpose([np.array([0, 0, kf])])
            self.H[:, i] = np.cross(np.reshape(self.B_P_Pi[i, :, :], (3, 3)), np.reshape(self.B_R_Pi[i, :, :], (3, 3))) * np.transpose(np.array([0, 0, kf])) + clock[i]*np.reshape(self.B_R_Pi[i, :, :], (3, 3)) * np.transpose(np.array([0, 0, kd]))

        Jb = np.concatenate((self.F, self.H), axis=0)
        JR = np.concatenate((np.concatenate((real_pose.orientation_matrix/self.m_tot, np.zeros((3, 3))), axis=0), np.concatenate((np.zeros((3, 3)), self.I_inv), axis=0)), axis=1)
        J = JR*Jb

        # Calcl u

        J_inv = np.linalg.inv(J)
        u = J_inv*(-f+v)
        for i in range(6):
            if u[i]<0:
                u[i] = 0
        desired_motor_speed = np.sqrt(u)

        return desired_motor_speed