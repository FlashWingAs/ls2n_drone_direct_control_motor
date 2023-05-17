import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import transforms3d as tf3d
from geometry_msgs.msg import Vector3Stamped
from ls2n_interfaces.msg import MotorControlSetPoint

from ls2n_drone_direct_motor_control.custom_common import Custom_Pose, Custom_Controller_Type, wedge_op

class Custom_Controller:

    def __init__(self, node):
        self.node = node
    
    type = Custom_Controller_Type.NONE
    desired_pose = Custom_Pose()

    def update_trajectory_setpoint(self, msg):
        for index, coordinate in enumerate(msg.joint_names):
            position = 0.0
            velocity = 0.0
            acceleration = 0.0
            if len(msg.points[0].positions) > index:
                position = msg.points[0].positions[index]
            if len(msg.points[0].velocities) > index:
                velocity = msg.points[0].velocities[index]
            if len(msg.points[0].accelerations) > index:
                acceleration = msg.points[0].accelerations[index]
            if coordinate == "x":
                self.desired_state.position[0] = position
                self.desired_state.velocity[0] = velocity
                self.desired_state.acceleration[0] = acceleration
            elif coordinate == "y":
                self.desired_state.position[1] = position
                self.desired_state.velocity[1] = velocity
                self.desired_state.acceleration[1] = acceleration
            elif coordinate == "z":
                self.desired_state.position[2] = position
                self.desired_state.velocity[2] = velocity
                self.desired_state.acceleration[2] = acceleration
            if coordinate == "phi":
                self.desired_state.rotation[0] = position
                self.desired_state.rot_velocity[0] = velocity
                self.desired_state.rot_acceleration[0] = acceleration
            elif coordinate == "theta":
                self.desired_state.rotation[1] = position
                self.desired_state.rot_velocity[1] = velocity
                self.desired_state.rot_acceleration[1] = acceleration
            elif coordinate == "psi":
                self.desired_state.rotation[2] = position
                self.desired_state.rot_velocity[2] = velocity
                self.desired_state.rot_acceleration[2] = acceleration
            elif coordinate == "R11" :
                self.desired_pose.rotation_matrix[0, 0] = position
                self.desired_pose.rotation_matrix_derivative[0, 0] = velocity
            elif coordinate == "R12" :
                self.desired_pose.rotation_matrix[0, 1] = position
                self.desired_pose.rotation_matrix_derivative[0, 1] = velocity
            elif coordinate == "R13" :
                self.desired_pose.rotation_matrix[0, 2] = position
                self.desired_pose.rotation_matrix_derivative[0, 2] = velocity
            elif coordinate == "R21" :
                self.desired_pose.rotation_matrix[1, 0] = position
                self.desired_pose.rotation_matrix_derivative[1, 0] = velocity
            elif coordinate == "R22" :
                self.desired_pose.rotation_matrix[1, 1] = position
                self.desired_pose.rotation_matrix_derivative[1, 1] = velocity
            elif coordinate == "R23" :
                self.desired_pose.rotation_matrix[1, 2] = position
                self.desired_pose.rotation_matrix_derivative[1, 2] = velocity
            elif coordinate == "R31" :
                self.desired_pose.rotation_matrix[2, 0] = position
                self.desired_pose.rotation_matrix_derivative[2, 0] = velocity
            elif coordinate == "R32" :
                self.desired_pose.rotation_matrix[2, 1] = position
                self.desired_pose.rotation_matrix_derivative[2, 1] = velocity
            elif coordinate == "R33" :
                self.desired_pose.rotation_matrix[2, 2] = position
                self.desired_pose.rotation_matrix_derivative[2, 2] = velocity
            else:
                self.node.get_logger.warn(
                    "Invalid coordinate" + coordinate + " received."
                )

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
    
    # Geometry
    pi = np.pi
    Lx = 0.686/2
    alpha = pi/3
    g = 9.81
    m_tot = 3.3981
    I1 = 0.1
    I2 = I1
    I3 = 0.15
    I = np.diag(np.array([I1, I2, I3]))
    I_inv = np.linalg.pinv(I)
    lambda_arms = np.array([-pi/2, pi/2, pi/6, -5*pi/6, -pi/6, 5*pi/6])
    alpha_arms = alpha*np.array([-1, 1, -1, 1, 1, -1])
    clock = np.array([1, -1, 1, -1, -1, 1]) # Clockwise or Anti-clockwise

    # Force and drag coefficient of the propeller -> replaced by computing control thrust-wise and using d as kd/kt
    # kf = 0.002
    # kd = 0.0002
    d = 0.06

    # PID coef
    k_p_pos = 4.0 #10.0
    k_i_pos = 0.0 #0.5
    k_d_pos = 4.0 #0.0001
    k_p_ang = k_p_pos #0.01
    k_i_ang = k_i_pos #0.001
    k_d_ang = k_d_pos #0.0
    yaw_p = 1.0
    yaw_i = 1.0
    yaw_d = 1.0
    geometric_controller_parameters = [k_p_pos, k_i_pos, k_d_pos, k_p_ang, k_i_ang, k_d_ang, yaw_p, yaw_i, yaw_d]
    Kpp = geometric_controller_parameters[0]*np.eye(3)
    Kpi = geometric_controller_parameters[1]*np.eye(3)
    Kpd = geometric_controller_parameters[2]*np.eye(3)
    Krp = geometric_controller_parameters[3]*np.eye(3)
    Kri = geometric_controller_parameters[4]*np.eye(3)
    Krd = geometric_controller_parameters[5]*np.eye(3)
    Krp[2, 2] = geometric_controller_parameters[3]*geometric_controller_parameters[6]
    Kri[2, 2] = geometric_controller_parameters[4]*geometric_controller_parameters[7]
    Krd[2, 2] = geometric_controller_parameters[5]*geometric_controller_parameters[8]

    # Errors init
    e_pos = np.zeros(3)
    D_e_pos = np.zeros(3)
    I_e_pos = np.zeros(3)

    e_ang = np.zeros(3)
    D_e_ang = np.zeros(3)
    I_e_ang = np.zeros(3)
    
    # Mixer init
    B_P_Pi = np.zeros((6,3,1))
    B_R_Pi = np.zeros((6,3,3))
    F = np.zeros((3,6))
    H = np.zeros((3,6))
    v_B = np.zeros(6)

    # Calc Jb
        
    for i in range(6):
        temp_lambda = R.from_euler('ZYX', np.array([lambda_arms[i], 0, 0])).as_matrix()
        temp_alpha = R.from_euler('ZYX', np.array([0, 0, alpha_arms[i]])).as_matrix()
        B_P_Pi[i, :, :] = np.matmul(temp_lambda, np.transpose([np.array([Lx, 0, 0])]))
        B_R_Pi[i, :, :] = np.matmul(temp_lambda, temp_alpha)
        F[:, [i]] = np.reshape(B_R_Pi[[i], :, 2], (3, 1))
        H[:, [i]] = np.cross(np.reshape(B_P_Pi[[i], :, :], (3, 1)),
                            np.reshape(B_R_Pi[[i], :, 2], (3, 1)),
                            axis=0) \
                    + clock[i]*d*np.reshape(B_R_Pi[[i], :, 2], (3, 1))

    Jb = np.concatenate((F, H), axis=0)


    def do_control(self, real_pose : Custom_Pose, step_size : float):

        # updates
        
        self.e_pos = self.desired_pose.position - real_pose.position
        self.I_e_pos = self.I_e_pos + self.e_pos*step_size
        self.D_e_pos = self.desired_pose.velocity - real_pose.velocity

        temp_e_ang = (real_pose.rotation.inverse)*(self.desired_pose.rotation)
        self.e_ang = 2*np.sign(temp_e_ang.scalar)*temp_e_ang.vector
        self.I_e_ang = self.I_e_ang + self.e_ang*step_size
        self.D_e_ang = self.desired_pose.rot_velocity - real_pose.rot_velocity
        # Calc V

        DD_p_d = self.desired_pose.acceleration
        DD_r_d = self.desired_pose.rot_acceleration

        Vp = DD_p_d + self.Kpd @ self.D_e_pos + self.Kpp @ self.e_pos + self.Kpi @ self.I_e_pos + np.array([0, 0, self.g])
        Vr = DD_r_d + self.Krd @ self.D_e_ang + self.Krp @ self.e_ang + self.Kri @ self.I_e_ang

        # Calc J

        f_B = real_pose.rotation.inverse.rotate(Vp*self.m_tot)
        tau_B = np.matmul(self.I, Vr)
        self.v_B = np.concatenate((f_B, tau_B))

        # Calc u

        u = np.matmul(np.linalg.inv(self.Jb), self.v_B)

        for i in range(6):
            if u[i]<0.05:
                u[i] = 0.05
        desired_motor_thrust = u

        return desired_motor_thrust