import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import transforms3d as tf3d
from geometry_msgs.msg import Vector3Stamped
from ls2n_interfaces.msg import MotorControlSetPoint

from ls2n_drone_tilthex_control.tilthex_common import Tilthex_Pose, Tilthex_Controller_Type, Tilthex_PID_Param, Tilthex_Observer

class Tilthex_Controller:

    def __init__(self, node):
        self.node = node
        self.type = Tilthex_Controller_Type.NONE
        self.desired_pose = Tilthex_Pose()

        # Geometry
        self.pi = np.pi
        self.Lx = 0.686/2
        self.alpha = self.pi/6
        self.g = 9.81
        self.m_tot = 3.3981
        self.I1 = 0.323
        self.I2 = self.I1
        self.I3 = 0.484
        self.I = np.diag(np.array([self.I1, self.I2, self.I3]))
        self.I_inv = np.linalg.pinv(self.I)
        self.lambda_arms = np.array([-self.pi/2, self.pi/2, self.pi/6, -5*self.pi/6, -self.pi/6, 5*self.pi/6])
        self.alpha_arms = self.alpha*np.array([-1, 1, -1, 1, 1, -1])
        self.clock = np.array([1, -1, 1, -1, -1, 1]) # Clockwise or Anti-clockwise
        self.d = 0.06 #0.06 #0.745

        # Mixer init
        self.B_P_Pi = np.zeros((6,3,1))
        self.B_R_Pi = np.zeros((6,3,3))
        self.F = np.zeros((3,6))
        self.H = np.zeros((3,6))
        self.v_B = np.zeros(6)
        for i in range(6):
            self.B_P_Pi[i, :, :] = self.Lx*np.transpose(np.array([[np.cos(self.lambda_arms[i]), np.sin(self.lambda_arms[i]), 0]]))
            self.B_R_Pi[i, :, :] = np.array([[np.cos(self.lambda_arms[i]), -np.sin(self.lambda_arms[i])*np.cos(self.alpha_arms[i]), np.sin(self.lambda_arms[i])*np.sin(self.alpha_arms[i])], 
                                             [np.sin(self.lambda_arms[i]), np.cos(self.lambda_arms[i])*np.cos(self.alpha_arms[i]), -np.cos(self.lambda_arms[i])*np.sin(self.alpha_arms[i])], 
                                             [0, np.sin(self.alpha_arms[i]), np.cos(self.alpha_arms[i])]])
            self.F[:, [i]] = np.reshape(self.B_R_Pi[[i], :, 2], (3, 1))
            self.H[:, [i]] = np.cross(np.reshape(self.B_P_Pi[[i], :, :], (3, 1)),
                                np.reshape(self.B_R_Pi[[i], :, 2], (3, 1)),
                                axis=0) \
                        + self.clock[i]*self.d*np.reshape(self.B_R_Pi[[i], :, 2], (3, 1))

        self.Jb = np.concatenate((self.F, self.H), axis=0)


    def update_trajectory_setpoint(self, msg):
        euler_vector = np.array([0.0, 0.0, 0.0])
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
                self.desired_pose.position[0] = position
                self.desired_pose.velocity[0] = velocity
                self.desired_pose.acceleration[0] = acceleration
            elif coordinate == "y":
                self.desired_pose.position[1] = position
                self.desired_pose.velocity[1] = velocity
                self.desired_pose.acceleration[1] = acceleration
            elif coordinate == "z":
                self.desired_pose.position[2] = position
                self.desired_pose.velocity[2] = velocity
                self.desired_pose.acceleration[2] = acceleration
            if coordinate == "roll":
                euler_vector[0] = position
                self.desired_pose.rot_velocity[0] = velocity*np.pi/180
                self.desired_pose.rot_acceleration[0] = acceleration*np.pi/180
            elif coordinate == "pitch":
                euler_vector[1] = position
                self.desired_pose.rot_velocity[1] = velocity*np.pi/180
                self.desired_pose.rot_acceleration[1] = acceleration*np.pi/180
            elif coordinate == "yaw":
                euler_vector[2] = position
                self.desired_pose.rot_velocity[2] = velocity*np.pi/180
                self.desired_pose.rot_acceleration[2] = acceleration*np.pi/180
            # else:
            #     self.node.get_logger().warn(
            #         "Invalid coordinate" + coordinate + " received."
            #     )
        self.desired_pose.rotation = Quaternion(np.roll(R.from_euler('XYZ', euler_vector, degrees = True).as_quat(), 1))

    def do_control(self, _):
        pass

    def integral_reset(self, _):
        pass

    def anti_windup(self, _):
        pass

    def joystick_input(self, _):
        pass

class Test_Controller(Tilthex_Controller):
    def __init__(self, node):
        super().__init__(node)
        self.type = Tilthex_Controller_Type.TEST

    def do_control(self):
        desired_motor_speed = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        return desired_motor_speed

class Geometric_Controller(Tilthex_Controller):

    def __init__(self, node):
        super().__init__(node)
        self.type = Tilthex_Controller_Type.GEOMETRIC
        
        self.default_pid_param_trans_p = np.array([15.0, 15.0, 10.0])
        self.default_pid_param_trans_i = np.array([2.0, 2.0, 2.0])
        self.default_pid_param_trans_d = np.array([15.0, 15.0, 15.0])
        self.default_pid_param_rot_p = np.array([10.0, 10.0, 10.0])
        self.default_pid_param_rot_i = np.array([1.0, 1.0, 0.5])
        self.default_pid_param_rot_d = np.array([4.0, 4.0, 4.0])
        self.PID = Tilthex_PID_Param(self.default_pid_param_trans_p, self.default_pid_param_trans_i, self.default_pid_param_trans_d,
                                    self.default_pid_param_rot_p, self.default_pid_param_rot_i, self.default_pid_param_rot_d)
        
        # Anti Windup
        self.anti_windup_trans_switch = False
        self.anti_windup_rot_switch = False
        self.anti_windup_trans = np.zeros(3)
        self.anti_windup_rot = np.zeros(3)

        # Errors init
        self.e_pos = np.zeros(3)
        self.D_e_pos = np.zeros(3)
        self.I_e_pos = np.zeros(3)

        self.e_ang = np.zeros(3)
        self.D_e_ang = np.zeros(3)
        self.I_e_ang = np.zeros(3)
        
        self.trans_increment = 0.2       # meters
        self.pitch_roll_increment = 2.0  # degrees
        self.yaw_increment = 5.0

        # Observer creation
        self.d_o_initialized = False
        self.d_o_wait = 0.0
        self.do_observer = False

    def integral_reset(self, trans = False, rot = False):
        if trans:
            self.I_e_pos = np.zeros(3)
        if rot:
            self.I_e_ang = np.zeros(3)

    def anti_windup(self, trans = False, rot = False):
        if trans:
            if self.I_e_pos[0] > self.anti_windup_trans[0]:
                self.I_e_pos[0] = self.anti_windup_trans[0]
            if self.I_e_pos[1] > self.anti_windup_trans[1]:
                self.I_e_pos[1] = self.anti_windup_trans[1]
            if self.I_e_pos[2] > self.anti_windup_trans[2]:
                self.I_e_pos[2] = self.anti_windup_trans[2]
        if rot:
            if self.I_e_ang[0] > self.anti_windup_rot[0]:
                self.I_e_ang[0] = self.anti_windup_rot[0]
            if self.I_e_ang[1] > self.anti_windup_rot[1]:
                self.I_e_ang[1] = self.anti_windup_rot[1]
            if self.I_e_ang[2] > self.anti_windup_rot[2]:
                self.I_e_ang[2] = self.anti_windup_rot[2]

    def joystick_input(self, axis, direction):
        string_axis = ['X', 'Y', 'Z']
        if axis <= 3:
            new_pose = self.desired_pose.position
            new_pose[axis - 1] += direction*self.trans_increment
            self.desired_pose.position = new_pose
            self.node.get_logger().info('Translation along ' + string_axis[axis -  1] + ' of ' + str(direction*self.trans_increment) + 'meters')
            self.node.get_logger().info('New desired position is ' + str(new_pose))
        else:
            axis -= 3
            new_pose = R.from_quat(np.roll(self.desired_pose.rotation.elements, -1)).as_euler('XYZ', degrees = True)
            if axis == 3:
                new_pose[axis - 1] += direction*self.yaw_increment
            else:
                new_pose[axis - 1] += direction*self.pitch_roll_increment
            self.desired_pose.rotation = Quaternion(np.roll(R.from_euler('XYZ', new_pose, degrees = True).as_quat(), 1))
            self.node.get_logger().info('Rotation about ' + string_axis[axis -  1] + ' of ' + str(direction*self.pitch_roll_increment) + 'degrees')
            self.node.get_logger().info('New desired orientiation is ' + str(new_pose))

    def translationnal_acceleration_check(self, Acc, real_pose : Tilthex_Pose):
        Ax = Acc[0]
        Ay = Acc[1]
        A_t = np.sqrt(Ax**2+Ay**2)
        Accel_direction = np.arctan2(Ay, Ax)
        P = self.g

        # SIMPLE HORIZONTAL SOLUTION (doesn't take into account the orientation of the drone)
        A_t_M_0 = P*np.tan(self.alpha)

        # Getting the simplified drone tilt angle
        rotation_matrix = R.from_quat(np.roll(real_pose.rotation.elements, -1)).as_matrix()
        z_b_0 = rotation_matrix[:, 2]
        theta = np.arccos(z_b_0[2])
        Tilt_direction = np.arctan2(z_b_0[1], z_b_0[0])

        # SIMPLE TILTED SOLUTION (take into account the orientation of the drone, but limit to the smallest acceleration possible)
        A_t_M_1 = P*np.tan(self.alpha - theta)

        # COMPLEX TILTED SOLUTION (take into account the orientation of the drone, 
        #                          and adapt the max acceleretation to the orientation direction and the translation direction)

        M = P*np.tan(self.alpha + theta)
        m = P*np.tan(self.alpha - theta)
        N = P*np.tan(self.alpha)

        a = (M + m)/2
        b = N
        l = b**2/a
        e = np.sqrt(1-b**2/a**2)
        def ellipse_radius(angle):
            return l/(1-e*np.cos(angle))
        A_t_M_2 = ellipse_radius(Accel_direction - Tilt_direction)

        A_t_M = [A_t_M_0, A_t_M_1, A_t_M_2]
        selection = 2

        if A_t > A_t_M[selection]:
            self.node.get_logger().info("tf_checker triggered", throttle_duration_sec = 1)
            ratio = A_t_M[selection] / A_t
            # Acc = Acc*ratio
            Ax *= ratio
            Ay *= ratio
            Acc = np.array([Ax, Ay, Acc[2]])

        return Acc

    def do_control(self, real_pose : Tilthex_Pose, step_size : float):

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

        # Anti Windup
        self.anti_windup(self.anti_windup_trans_switch, self.anti_windup_rot_switch)

        self.Vp = DD_p_d + self.PID.Ktd @ self.D_e_pos + self.PID.Ktp @ self.e_pos + self.PID.Kti @ self.I_e_pos + np.array([0, 0, self.g])
        self.Vp = self.translationnal_acceleration_check(self.Vp, real_pose)
        self.Vr = DD_r_d + self.PID.Krd @ self.D_e_ang + self.PID.Krp @ self.e_ang + self.PID.Kri @ self.I_e_ang

        # Calc J
        f_B = real_pose.rotation.inverse.rotate(self.Vp*self.m_tot)
        tau_B = np.matmul(self.I, self.Vr)
        self.v_B = np.concatenate((f_B, tau_B))
        first_time = False

        if self.do_observer and not self.d_o_initialized:
            self.disturbance_observer = Tilthex_Observer(self.node, self.m_tot, self.I)
            self.d_o_initialized = True
            first_time = True
            self.d_o_wait = self.node.time
            self.node.get_logger().info("Disturbance Observer Initialised")
        if self.d_o_initialized:
            if self.do_observer:
                if self.node.time - self.d_o_wait >= 10.0:
                    if first_time:
                        self.node.get_logger().info("Disturbance Observer Active")
                        first_time = False
                    self.v_B -= self.disturbance_observer.estimated_wrench
                self.disturbance_observer.do_observer(real_pose, self.v_B, self.node.time)
                
                

        # Calc u
        u = np.matmul(np.linalg.inv(self.Jb), self.v_B)

        for i in range(6):
            if u[i]<0.0:
                u[i] = 0.0
        desired_motor_thrust = u

        return desired_motor_thrust
    
class Velocity_Controller(Tilthex_Controller):
    
    def __init__(self, node):
        super().__init__(node)
        self.type = Tilthex_Controller_Type.VELOCITY

        self.default_pid_param_trans_p = np.array([2.0, 2.0, 2.0])
        self.default_pid_param_trans_i = np.array([0.5, 0.5, 0.5])
        self.default_pid_param_trans_d = np.array([0.0, 0.0, 0.0])
        self.default_pid_param_rot_p = np.array([2.0, 2.0, 2.0])
        self.default_pid_param_rot_i = np.array([0.5, 0.5, 0.5])
        self.default_pid_param_rot_d = np.array([0.0, 0.0, 0.0])
        self.PID = Tilthex_PID_Param(self.default_pid_param_trans_p, self.default_pid_param_trans_i, self.default_pid_param_trans_d,
                                    self.default_pid_param_rot_p, self.default_pid_param_rot_i, self.default_pid_param_rot_d)

        # Errors init
        self.e_pos = np.zeros(3)
        self.I_e_pos = np.zeros(3)

        self.e_ang = np.zeros(3)
        self.I_e_ang = np.zeros(3)

        self.trans_increment = 0.01       # meters/s
        self.pitch_roll_increment = 0.02  # degrees/s
        self.yaw_increment = 0.05

    def integral_reset(self, trans = False, rot = False):
        if trans:
            self.I_e_pos = np.zeros(3)
        if rot:
            self.I_e_ang = np.zeros(3)

    def joystick_input(self, axis, direction):
        string_axis = ['X', 'Y', 'Z']
        if axis <= 3:
            new_pose = self.desired_pose.velocity
            new_pose[axis - 1] += direction*self.trans_increment
            self.desired_pose.velocity = new_pose
            self.node.get_logger().info('Velocity along ' + string_axis[axis -  1] + ' of ' + str(direction*self.trans_increment*100) + 'centimeters/seconds')
            self.node.get_logger().info('New desired speed is ' + str(new_pose))
        else:
            axis -= 3
            new_pose = self.desired_pose.rot_velocity
            if axis == 3:
                new_pose[axis - 1] += direction*self.yaw_increment
            else:
                new_pose[axis - 1] += direction*self.pitch_roll_increment
            self.desired_pose.rot_velocity = new_pose
            self.node.get_logger().info('Rotation velocity about ' + string_axis[axis -  1] + ' of ' + str(direction*self.pitch_roll_increment*100) + 'centidegrees/seconds')
            self.node.get_logger().info('New desired orientiation velocity is ' + str(new_pose))

    def do_control(self, real_pose : Tilthex_Pose, step_size : float):

        # updates
        self.e_pos = self.desired_pose.velocity - real_pose.velocity
        self.I_e_pos = self.I_e_pos + self.e_pos*step_size

        self.e_ang = self.desired_pose.rot_velocity - real_pose.rot_velocity
        self.I_e_ang = self.I_e_ang + self.e_ang*step_size

        self.Vp = self.PID.Ktp @ self.e_pos + self.PID.Kti @ self.I_e_pos + np.array([0, 0, self.g])
        self.Vr = self.PID.Krp @ self.e_ang + self.PID.Kri @ self.I_e_ang

        # Calc J
        f_B = real_pose.rotation.inverse.rotate(self.Vp*self.m_tot)
        tau_B = np.matmul(self.I, self.Vr)
        self.v_B = np.concatenate((f_B, tau_B))

        # Calc u
        u = np.matmul(np.linalg.inv(self.Jb), self.v_B)

        for i in range(6):
            if u[i]<0.0:
                u[i] = 0.0
        desired_motor_thrust = u

        return desired_motor_thrust