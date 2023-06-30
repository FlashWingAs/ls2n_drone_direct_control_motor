import importlib
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import os
import subprocess
import signal
import rclpy
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Joy as JoyMsg
from ls2n_interfaces.msg import (
    AttitudeThrustSetPoint,
    DroneStatus,
    RatesThrustSetPoint,
    KeepAlive,
    MotorControlSetPoint,
    TilthexDebug,
    TilthexPoseDebug,
)
from ls2n_interfaces.srv import DroneRequest, OnboardApp
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_srvs.srv import Empty

from ls2n_drone_bridge.common import (
    DroneRequestString,
    DroneStatusString,
    FullState,
    qos_profile_sensor_data,
)
from ls2n_drone_tilthex_control.tilthex_controllers import (
    Test_Controller,
    Geometric_Controller,
    Velocity_Controller
)
from ls2n_drone_tilthex_control.tilthex_common import (
    Tilthex_Controller_Type,
    Tilthex_Pose,
    Tilthex_PID_Param,
)
from ls2n_drone_bridge.drone_bridge import (
    Status
)

class TilthexControlCenter(Node):
    def __init__(self):
        super().__init__("Tilthex_Control")
        parameters = [
            [
                "mass",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Drone mass"
            ],
            [
                "max_thrust",
                10000.0,
                ParameterType.PARAMETER_DOUBLE,
                "Total drone max thrust"
            ],
            [
                "select_controller",
                2,
                ParameterType.PARAMETER_INTEGER,
                "Tilthex controller selection"
            ],
            [
                "geom_pid_trans_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "geom_pid_trans_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "geom_pid_trans_d",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "geom_pid_rot_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "geom_pid_rot_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "geom_pid_rot_d",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "vel_pid_trans_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "vel_pid_trans_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "vel_pid_rot_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "vel_pid_rot_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "d_position",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Desired position"
            ],
            [
                "d_rot_euler",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Desired orientation expressed as Euler angles"
            ],
            [
                "anti_windup_trans",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Anti Windup parameter for translation integral part"
            ],
            [
                "anti_windup_rot",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Anti Windup parameter for rotation integral part"
            ],
            [
                "bat_discharge_a",
                0.0,
                ParameterType.PARAMETER_DOUBLE,
                "Battery discharge model parameter a"
            ],
            [
                "bat_discharge_b",
                0.0,
                ParameterType.PARAMETER_DOUBLE,
                "Battery discharge model parameter b"
            ],
        ]
        for parameter in parameters:
            self.declare_parameter(
                parameter[0],
                parameter[1],
                ParameterDescriptor(
                    type=parameter[2], description=parameter[3]
                ),
            )
            setattr(
                self,
                parameter[0],
                lambda param=parameter[0]: self.get_parameter(param).value,
            )
        
        # Automatic detection of custom parameters

        self.custom_geom_pid_switch = False                   # True if using default pid parameters set in the controller
        if self.geom_pid_trans_p() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        if self.geom_pid_trans_i() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        if self.geom_pid_trans_d() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        if self.geom_pid_rot_p() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        if self.geom_pid_rot_i() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        if self.geom_pid_rot_d() != "0.0, 0.0, 0.0":
            self.custom_geom_pid_switch = True
        self.custom_vel_pid_switch = False                   # True if using default pid parameters set in the controller
        if self.vel_pid_trans_p() != "0.0, 0.0, 0.0":
            self.custom_vel_pid_switch = True
        if self.vel_pid_trans_i() != "0.0, 0.0, 0.0":
            self.custom_vel_pid_switch = True
        if self.vel_pid_rot_p() != "0.0, 0.0, 0.0":
            self.custom_vel_pid_switch = True
        if self.vel_pid_rot_i() != "0.0, 0.0, 0.0":
            self.custom_vel_pid_switch = True
        self.anti_windup_trans_switch = False
        self.anti_windup_rot_switch = False
        if self.anti_windup_trans() != "0.0, 0.0, 0.0":
            self.anti_windup_trans_switch = True
        if self.anti_windup_rot() != "0.0, 0.0, 0.0":
            self.anti_windup_rot_switch = True

        self.get_logger().info("Starting Tilthex Control node")
        # Service creation
        self.create_service(Empty, "SpinMotors", self.spin_motors)
        self.create_service(Empty, "StartExperiment", self.start_experiment)
        self.create_service(Empty, "StopExperiment", self.stop_experiment)
        self.drone_request_client = self.create_client(
            DroneRequest,
            "Request",
        )
        self.start_trajectory_client = self.create_client(
            Empty, "StartTrajectory"
        )
        self.reset_trajectory_client = self.create_client(
            Empty, "ResetTrajectory"
        )
        
        # Bridge feedback
        self.create_subscription(
            DroneStatus,
            "Status",
            self.status_update_callback,
            qos_profile_sensor_data,
        )


        self.create_subscription(
            Odometry,
            "EKF/odom",
            self.odom_update_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            JointTrajectory,
            "Trajectory",
            self.trajectory_callback,
            qos_profile_sensor_data,
        )
        
        # Set points
        self.direct_motor_control_publisher = self.create_publisher(
            MotorControlSetPoint,
            "MotorControlSetPoint",
            qos_profile_sensor_data
        )

        # Debug Topics
        self.classic_debug_publisher = self.create_publisher(
            TilthexDebug,
            "classic_debug",
            qos_profile_sensor_data
        )

        self.real_pose_debug_publisher = self.create_publisher(
            TilthexPoseDebug,
            "real_pose",
            qos_profile_sensor_data
        )

        self.desired_pose_debug_publisher = self.create_publisher(
            TilthexPoseDebug,
            "desired_pose",
            qos_profile_sensor_data
        )

        self.take_off_pose_debug_publisher = self.create_publisher(
            TilthexPoseDebug,
            "take_off_pose",
            qos_profile_sensor_data
        )

        self.joy_sub = self.create_subscription(
            JoyMsg,
            "/CommandCenter/Joystick",
            self.joystick_callback,
            qos_profile_sensor_data,
        )
        
        # subscribers related to disturbances observation
        # self.create_subscription(
        #     Vector3Stamped,
        #     "Observer/DisturbancesWorld",
        #     self.disturbances_callback,
        #     qos_profile_sensor_data,
        #)

        self.main_loop_timer = self.create_timer(
            0.01, self.main_loop
            )
        
        self.battery_loop_timer = self.create_timer(
            2.0, self.battery_loop
            )
        
        self.add_on_set_parameters_callback(self.parameters_callback)

    status = Status()
    experiment_started = False
    odom = Odometry()
    real_pose = Tilthex_Pose()
    take_off_pose = real_pose.copy()
    land_pose = real_pose.copy()
    controller = None
    sec = 0
    nanosec = 0.0
    time = 0.0
    time_prev = 0.0
    step_size = 0.0
    step_size_old = 0.0
    taking_off_flag = False
    landing_flag = False
    flying_flag = False
  
    # Callbacks

    def parameters_callback(self, params):
        for param in params:
            if self.controller is not None and self.controller.type != Tilthex_Controller_Type.TEST:
                if param.name == "select_controller":
                        self.controller_selection(self.select_controller(), init = True, reset = True)
                if self.controller.type == Tilthex_Controller_Type.GEOMETRIC:
                    if param.name == "geom_pid_trans_p":
                        self.controller.PID.trans_p = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "geom_pid_trans_i":
                        self.controller.PID.trans_i = np.fromstring(param.value, sep = ",")
                        self.controller.integral_reset(trans = True)
                        self.controller.PID.update()
                    if param.name == "geom_pid_trans_d":
                        self.controller.PID.trans_d = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "geom_pid_rot_p":
                        self.controller.PID.rot_p = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "geom_pid_rot_i":
                        self.controller.PID.rot_i = np.fromstring(param.value, sep = ",")
                        self.controller.integral_reset(rot = True)
                        self.controller.PID.update()
                    if param.name == "geom_pid_rot_d":
                        self.controller.PID.rot_d = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                if self.controller.type == Tilthex_Controller_Type.VELOCITY:
                    if param.name == "vel_pid_trans_p":
                        self.controller.PID.trans_p = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "vel_pid_trans_i":
                        self.controller.PID.trans_i = np.fromstring(param.value, sep = ",")
                        self.controller.integral_reset(trans = True)
                        self.controller.PID.update()
                    if param.name == "vel_pid_trans_d":
                        self.controller.PID.trans_d = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "vel_pid_rot_p":
                        self.controller.PID.rot_p = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                    if param.name == "vel_pid_rot_i":
                        self.controller.PID.rot_i = np.fromstring(param.value, sep = ",")
                        self.controller.integral_reset(rot = True)
                        self.controller.PID.update()
                    if param.name == "vel_pid_rot_d":
                        self.controller.PID.rot_d = np.fromstring(param.value, sep = ",")
                        self.controller.PID.update()
                if param.name == "d_position":
                    self.controller.desired_pose.position = np.fromstring(param.value, sep = ',')
                if param.name == "d_rot_euler":
                    self.controller.desired_pose.rotation = Quaternion(np.roll(R.from_euler('XYZ', np.fromstring(param.value, sep = ','), degrees = True).as_quat(), 1))
                if param.name == "anti_windup_trans":
                    self.controller.anti_windup_trans = np.fromstring(param.value, sep = ',')
                    if self.anti_windup_trans == "0.0, 0.0, 0.0":
                        self.anti_windup_trans_switch = False
                        self.get_logger().info("Translationnal anti-windup deactivated")
                    else: 
                        self.anti_windup_trans_switch = True
                        self.get_logger().info("Translationnal anti-windup activated")
                if param.name == "anti_windup_rot":
                    self.controller.anti_windup_rot = np.fromstring(param.value, sep = ',')
                    if self.anti_windup_rot == "0.0, 0.0, 0.0":
                        self.anti_windup_rot_switch = False
                        self.get_logger().info("Rotationnal anti-windup deactivated")
                    else: 
                        self.anti_windup_rot_switch = True
                        self.get_logger().info("Rotationnal anti-windup activated")
                
                self.get_logger().info("Parameter succesfully updated live")
            else:
                self.get_logger().info("Parameter updated, will be applied next controller initialisation")
        return SetParametersResult(successful=True)

    def status_update_callback(self, rcvd_msg):
        self.status.status = rcvd_msg.status
        if (self.status.status == DroneStatus.FLYING and not self.experiment_started):
            self.experiment_started = True
        if (self.status.status == DroneStatus.IDLE and self.experiment_started):
            self.experiment_started = False

    def odom_update_callback(self, odom):
        self.odom = odom
        self.controller_execute()
        
    def trajectory_callback(self, msg):
        if self.controller is not None:
            if self.controller.type == Tilthex_Controller_Type.GEOMETRIC:
                if self.status.status == DroneStatus.FLYING and self.experiment_started:
                    self.controller.update_trajectory_setpoint(msg)

    TX_ref = 1
    TY_ref = 0
    TZ_ref = 4
    RX_ref = 6
    RY_ref = 7
    RZ_ref = 3
    L3_ref = 9
    R3_ref = 10
    RB_ref = 5
    LB_ref = 4
    START_ref = 7
    SELECT_ref = 6
    TX = 0
    TY = 0
    TZ = 0
    RX = 0
    RY = 0
    RZ = 0
    L3 = 0
    R3 = 0
    RB = 0
    LB = 0
    START = 0
    SELECT = 0

    def joystick_callback(self, data):
        tx = np.round(data.axes[self.TX_ref], 1)
        ty = np.round(data.axes[self.TY_ref], 1)
        tz = np.round(data.axes[self.TZ_ref], 1)
        rx = np.round(data.axes[self.RX_ref], 1)
        ry = np.round(data.axes[self.RY_ref], 1)
        rz = np.round(data.axes[self.RZ_ref], 1)
        l3 = np.round(data.buttons[self.L3_ref], 1)
        r3 = np.round(data.buttons[self.R3_ref], 1)
        rb = np.round(data.buttons[self.RB_ref], 1)
        lb = np.round(data.buttons[self.LB_ref], 1)
        start = np.round(data.buttons[self.START_ref], 1)
        select = np.round(data.buttons[self.SELECT_ref], 1)
        if self.experiment_started:

            if (abs(tx) >= 0.5) and self.TX == 0:
                self.TX = np.sign(tx)
                if self.TX > 0 :
                    self.controller.joystick_input(1, 1)
                elif self.TX < 0:
                    self.controller.joystick_input(1, -1)
            if (abs(tx) < 0.5) and self.TX != 0:
                self.TX = 0

            if (abs(ty) >= 0.5) and self.TY == 0:
                self.TY = np.sign(ty)
                if self.TY > 0 :
                    self.controller.joystick_input(2, 1)
                elif self.TY < 0:
                    self.controller.joystick_input(2, -1)
            if (abs(ty) < 0.5) and self.TY != 0:
                self.TY = 0
            
            if (abs(tz) >= 0.5) and self.TZ == 0:
                self.TZ = np.sign(tz)
                if self.TZ > 0 :
                    self.controller.joystick_input(3, 1)
                elif self.TZ < 0:
                    self.controller.joystick_input(3, -1)
            if (abs(tz) < 0.5) and self.TZ != 0:
                self.TZ = 0

            if (abs(rx) >= 0.5) and self.RX == 0:
                self.RX = np.sign(rx)
                if self.RX > 0 :
                    self.controller.joystick_input(4, -1)
                elif self.RX < 0:
                    self.controller.joystick_input(4, 1)
            if (abs(rx) < 0.5) and self.RX != 0:
                self.RX = 0

            if (abs(ry) >= 0.5) and self.RY == 0:
                self.RY = np.sign(ry)
                if self.RY > 0 :
                    self.controller.joystick_input(5, 1)
                elif self.RY < 0:
                    self.controller.joystick_input(5, -1)
            if (abs(ry) < 0.5) and self.RY != 0:
                self.RY = 0

            if (abs(rz) >= 0.5) and self.RZ == 0:
                self.RZ = np.sign(rz)
                if self.RZ > 0 :
                    self.controller.joystick_input(6, 1)
                elif self.RZ < 0:
                    self.controller.joystick_input(6, -1)
            if (abs(rz) < 0.5) and self.RZ != 0:
                self.RZ = 0

            if (abs(l3 >= 0.5)) and self.L3 == 0:
                self.L3 = 1
                new_pose = Tilthex_Pose()
                new_pose.position = np.fromstring(self.d_position(), sep=",")
                new_pose.rotation = self.controller.desired_pose.rotation
                self.controller.desired_pose = new_pose
                self.get_logger().info('Translation reset to ' + str(new_pose.position))
            if (abs(l3 < 0.5) and self.L3) != 0:
                self.L3 = 0

            if (abs(r3 >= 0.5)) and self.R3 == 0:
                self.R3 = 1
                new_pose = Tilthex_Pose()
                new_pose.rotation = Quaternion()
                new_pose.position = self.controller.desired_pose.position
                self.controller.desired_pose = new_pose
                self.get_logger().info('Rotation reset to ' + str(new_pose.rotation))
            if (abs(r3 < 0.5) and self.R3) != 0:
                self.R3 = 0
            
            if (abs(start >= 0.5)) and self.START == 0:
                self.START = 1
                self.start_trajectory_client.call_async(Empty.Request())
                self.get_logger().info('Starting trajectory')
            if (abs(start < 0.5) and self.START) != 0:
                self.START = 0

            if (abs(select >= 0.5)) and self.SELECT == 0:
                self.SELECT = 1
                self.reset_trajectory_client.call_async(Empty.Request())
                new_pose = Tilthex_Pose()
                new_pose.position = np.fromstring(self.d_position(), sep=",")
                new_pose.rotation = Quaternion()
                self.controller.desired_pose = new_pose
                self.get_logger().info('Stopping and reseting trajectory')
            if (abs(select < 0.5)) and self.SELECT != 0:
                self.SELECT = 0

            if (abs(rb >= 0.5)) and self.RB == 0:
                self.RB = 1
                self.controller_selection(2, init = True, reset = True)
            if (abs(rb < 0.5)) and self.RB != 0:
                self.RB = 0

            if (abs(lb >= 0.5)) and self.LB == 0:
                self.LB = 1
                self.controller_selection(3, init = True, reset = True)
            if (abs(lb < 0.5)) and self.LB != 0:
                self.LB = 0

    # Publishers

    def direct_motor_control_transfer(self, motors_set_points):
        to_send = np.zeros(12)
        max = 0.0
        for i in range(len(motors_set_points)):
            abs = self.thrust2abs(motors_set_points[i])
            if abs > max:
                max = abs
            if abs < 0.05:
                abs = 0.05
            to_send[i] = abs
        if max > 1:
            to_send = to_send / max
        to_send_L = to_send.tolist()
        msg = MotorControlSetPoint()
        msg.motor_velocity = to_send_L
        self.direct_motor_control_publisher.publish(msg)

    def classic_debug(self):
        msg = TilthexDebug()
        msg.step_size = float(self.step_size)
        msg.v_b = self.controller.v_B
        msg.v_p = self.controller.Vp
        msg.v_r = self.controller.Vr
        msg.current_position = self.real_pose.position.tolist()
        msg.current_rotation = np.concatenate((np.array([self.real_pose.rotation.scalar]), self.real_pose.rotation.vector)).tolist()
        msg.goal_position = self.controller.desired_pose.position.tolist()
        msg.goal_rotation = np.concatenate((np.array([self.controller.desired_pose.rotation.scalar]), self.controller.desired_pose.rotation.vector)).tolist()
        msg.error_position = self.controller.e_pos.tolist()
        msg.error_rotation = self.controller.e_ang.tolist()
        msg.integral_pos = self.controller.I_e_pos.tolist()
        if self.controller.d_o_initialized:
            msg.observer = self.controller.disturbance_observer.estimated_wrench.tolist()
        else:
            msg.observer = [0, 0, 0, 0 ,0, 0]
        self.classic_debug_publisher.publish(msg)

    def real_pose_debug(self):
        msg = TilthexPoseDebug()
        msg.position = self.real_pose.position.tolist()
        msg.velocity = self.real_pose.velocity.tolist()
        msg.acceleration = self.real_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.real_pose.rotation.scalar]), self.real_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.real_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.real_pose.rot_acceleration.tolist()
        self.real_pose_debug_publisher.publish(msg)

    def desired_pose_debug(self):
        msg = TilthexPoseDebug()
        msg.position = self.controller.desired_pose.position.tolist()
        msg.velocity = self.controller.desired_pose.velocity.tolist()
        msg.acceleration = self.controller.desired_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.controller.desired_pose.rotation.scalar]), self.controller.desired_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.controller.desired_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.controller.desired_pose.rot_acceleration.tolist()
        self.desired_pose_debug_publisher.publish(msg)

    def take_off_pose_debug(self):
        msg = TilthexPoseDebug()
        msg.position = self.take_off_pose.position.tolist()
        msg.velocity = self.take_off_pose.velocity.tolist()
        msg.acceleration = self.take_off_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.take_off_pose.rotation.scalar]), self.take_off_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.take_off_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.take_off_pose.rot_acceleration.tolist()
        self.take_off_pose_debug_publisher.publish(msg)

    # Services
    def init_controller(self):
        if self.controller.type == Tilthex_Controller_Type.TEST:
            pass
        if self.controller.type == Tilthex_Controller_Type.GEOMETRIC:
            self.controller.m_tot = self.mass()
            self.desired_altitude = np.fromstring(self.d_position(), sep=",")[2]
            if self.custom_geom_pid_switch:
                self.controller.PID = Tilthex_PID_Param(np.fromstring(self.geom_pid_trans_p(), sep = ','),
                                                       np.fromstring(self.geom_pid_trans_i(), sep = ','),
                                                       np.fromstring(self.geom_pid_trans_d(), sep = ','),
                                                       np.fromstring(self.geom_pid_rot_p(), sep = ','),
                                                       np.fromstring(self.geom_pid_rot_i(), sep = ','),
                                                       np.fromstring(self.geom_pid_rot_d(), sep = ','))
                self.get_logger().info("Initilised wit custom PID parameters", throttle_duration_sec = 1)
            else:
                self.get_logger().info("Initilised with default PID parameters", throttle_duration_sec = 1)
            if self.anti_windup_trans_switch:
                self.controller.anti_windup_trans = np.fromstring(self.anti_windup_trans(), sep = ',')
                self.get_logger().info("Translationnal anti windup activated", throttle_duration_sec = 1)
            if self.anti_windup_rot_switch:
                self.controller.anti_windup_rot = np.fromstring(self.anti_windup_rot(), sep = ',')
                self.get_logger().info("Rotationnal anti windup activated", throttle_duration_sec = 1)
            new_pose = Tilthex_Pose()
            new_pose.position = self.real_pose.position
            self.controller.desired_pose = new_pose
        if self.controller.type == Tilthex_Controller_Type.VELOCITY:
            self.get_logger().info("VELOCITY controller will be available in-flight. GEOMETRIC controller is used to take off and land", throttle_duration_sec = 1)
            self.controller.m_tot = self.mass()
            if self.custom_vel_pid_switch:
                self.controller.PID = Tilthex_PID_Param(np.fromstring(self.vel_pid_trans_p(), sep = ','),
                                                       np.fromstring(self.vel_pid_trans_i(), sep = ','),
                                                       np.zeros(3),
                                                       np.fromstring(self.vel_pid_rot_p(), sep = ','),
                                                       np.fromstring(self.vel_pid_rot_i(), sep = ','),
                                                       np.zeros(3))
                self.get_logger().info("Initilised wit custom PID parameters", throttle_duration_sec = 1)
            else:
                self.get_logger().info("Initilised with default PID parameters", throttle_duration_sec = 1)

    def spin_motors(self, request, response):
        if not self.experiment_started:
            self.controller_selection(self.select_controller(), init = True, reset = True)
            request_out = DroneRequest.Request()
            request_out.request = DroneRequest.Request.SPIN_MOTORS
            self.drone_request_client.call_async(request_out)
        else:
            self.get_logger().info(
                "Experiment already started. Stop experiment to reset."
            )
        return response

    def start_experiment(self, request, response):
        if not self.experiment_started:
            if self.status.status == DroneStatus.ARMED:
                request_out = DroneRequest.Request()
                request_out.request = DroneRequest.Request.TILTHEX_CONTROL
                self.drone_request_client.call_async(request_out)

                # Starting the controller with take off procedure
                self.get_logger().info("Experiment starting")
                self.switch_take_off()
                # Enabling the controller
                self.experiment_started = True
            else:
                self.get_logger().info(
                    "Drone not armed. Arm the drone then start again."
                )
        else:
            self.get_logger().info(
                "Experiment already started. Stop experiment to reset."
            )
        return response
    
    def stop_experiment(self, request, response):
        if self.experiment_started:
            self.get_logger().info("Experiment stopping")
            self.switch_land()
        else:
            self.get_logger().info(
                "Experiment already stopped."
            )
        return response

    # Translations

    def odom2tilthex_pose(self, odometry):
        custom = Tilthex_Pose()
        custom.position[0] = odometry.pose.pose.position.x
        custom.position[1] = odometry.pose.pose.position.y
        custom.position[2] = odometry.pose.pose.position.z
        custom.velocity[0] = odometry.twist.twist.linear.x
        custom.velocity[1] = odometry.twist.twist.linear.y
        custom.velocity[2] = odometry.twist.twist.linear.z
        custom.rotation = Quaternion(odometry.pose.pose.orientation.w, odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z)
        custom.rot_velocity[0] = odometry.twist.twist.angular.x
        custom.rot_velocity[1] = odometry.twist.twist.angular.y
        custom.rot_velocity[2] = odometry.twist.twist.angular.z
        return custom

    def thrust2abs(self, thrust: float) -> float:
        abs = thrust/(self.max_computed_thrust/6.0)
        return abs

    # State machine

    selected_controller = 0

    def controller_selection(self, selection, init = True, reset = False):
        do_init = False
        if reset: do_init = True
        if init and self.selected_controller != selection: do_init = True
        match selection:
            case 1:
                self.selected_controller = 1
                self.controller = Test_Controller(self)
                self.get_logger().info("Experiment starting with TEST controller", throttle_duration_sec = 1)
            case 2:
                self.selected_controller = 2
                self.controller = Geometric_Controller(self)
                self.get_logger().info("Experiment starting with GEOMETRIC controller", throttle_duration_sec = 1)
            case 3:
                self.selected_controller = 3
                self.controller = Velocity_Controller(self)
                self.get_logger().info("Experiment starting with VELOCITY controller", throttle_duration_sec = 1)
        if do_init:
            self.init_controller()

    def switch_take_off(self):
        if self.controller.type == Tilthex_Controller_Type.VELOCITY:
            self.controller = Geometric_Controller(self)
            self.init_controller()
        self.take_off_pose.position = self.real_pose.position
        self.take_off_pose.position[2] = self.desired_altitude
        self.controller.desired_pose = self.take_off_pose
        self.taking_off_flag = True
        self.get_logger().info("Drone taking off")

    def switch_flight(self):
        if self.selected_controller == 3:
            self.controller = Velocity_Controller(self)
            self.init_controller()
            self.get_logger().info("Take off complete")
        else:
            self.get_logger().info("Take off complete, heading to desired position")
            self.controller.desired_pose.position = np.fromstring(self.d_position(), sep=",")
            self.controller.desired_pose.rotation = Quaternion()
            self.flying_flag = True

    def switch_land(self):
        if self.controller.type == Tilthex_Controller_Type.VELOCITY:
            self.controller = Geometric_Controller(self)
        self.land_pose.position = self.real_pose.position
        self.land_pose.position[2] = 0.3
        self.controller.desired_pose = self.land_pose
        self.landing_flag = True
        self.get_logger().info("Drone landing")

    def switch_landed(self):
        request_out = DroneRequest.Request()
        request_out.request = DroneRequest.Request.DISARM
        self.drone_request_client.call_async(request_out)
        self.experiment_started = False
        self.controller = None
        self.get_logger().info("Landing complete,\nExperiment stopped")

    def taking_off(self):
        error = np.linalg.norm(self.take_off_pose.position - self.real_pose.position)
        if error < 0.1:
            self.taking_off_flag = False
            self.switch_flight()

    def landing(self):
        error = np.linalg.norm(self.land_pose.position - self.real_pose.position)
        if error < 0.15 and np.linalg.norm(self.real_pose.velocity) < 0.15:
            self.landing_flag = False
            self.switch_landed()

    t_start_hover = 0.0
    waiting_for_d_o = False

    def flying(self):
        error = np.linalg.norm(self.controller.desired_pose.position - self.real_pose.position)
        if error < 0.05 and self.waiting_for_d_o is False:
            self.t_start_hover = self.time
            self.waiting_for_d_o = True    
        if self.time - self.t_start_hover >= 2.0 and self.waiting_for_d_o:
            self.flying_flag = False
            if self.selected_controller == 2:
                self.controller.do_observer = True
            self.get_logger().info("Drone hovering in default pose")

    # Main loop

    def main_loop(self):
        pass
    
    max_computed_thrust = 10000.0
    bat_v = 0.0

    def battery_loop(self):
        self.max_computed_thrust = self.max_thrust() - (self. bat_discharge_a()*self.bat_v + self.bat_discharge_b())

    # Special actions


    vsn_limits_x = [-1.0, 1.0]
    vsn_limits_y = [-2.0, 2.0]
    vsn_limits_z = [0.0, 2.7]
    safe_return = 0.2
    
    def virtual_safety_net(self):
        vsn_trigger = False
        if self.controller.desired_pose.position[0] < self.vsn_limits_x[0] or self.real_pose.position[0] < self.vsn_limits_x[0]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[0] = self.vsn_limits_x[0] + self.safe_return
        if self.controller.desired_pose.position[0] > self.vsn_limits_x[1] or self.real_pose.position[0] > self.vsn_limits_x[1]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[0] = self.vsn_limits_x[1] - self.safe_return
        if self.controller.desired_pose.position[1] < self.vsn_limits_y[0] or self.real_pose.position[1] < self.vsn_limits_y[0]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[1] = self.vsn_limits_y[0] + self.safe_return
        if self.controller.desired_pose.position[1] > self.vsn_limits_y[1] or self.real_pose.position[1] > self.vsn_limits_y[1]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[1] = self.vsn_limits_y[1] - self.safe_return
        if self.controller.desired_pose.position[2] < self.vsn_limits_z[0] or self.real_pose.position[2] < self.vsn_limits_z[0]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[2] = self.vsn_limits_z[0] + self.safe_return
        if self.controller.desired_pose.position[2] > self.vsn_limits_z[1] or self.real_pose.position[2] > self.vsn_limits_z[1]:
            vsn_trigger = True
            new_pose = np.copy(self.real_pose.position)
            new_pose[2] = self.vsn_limits_z[1] - self.safe_return
        if vsn_trigger == True:
            self.get_logger().info("Virtual Safety Net triggered, attempting return to initial pose", throttle_duration_sec = 1)
            if self.selected_controller != 2:
                self.controller_selection(2, init = True, reset = False)
            self.reset_trajectory_client.call_async(Empty.Request())
            self.controller.desired_pose.position = new_pose
            self.controller.desired_pose.rotation = Quaternion()

    rot_limits = 15.0*np.pi/180

    def rotation_security(self):
        rot_trigger = False
        real_rotation_matrix = R.from_quat(np.roll(self.real_pose.rotation.elements, -1)).as_matrix()
        real_z_b_0 = real_rotation_matrix[:, 2]
        real_theta = np.arccos(real_z_b_0[2])
        desired_rotation_matrix = R.from_quat(np.roll(self.controller.desired_pose.rotation.elements, -1)).as_matrix()
        desired_z_b_0 = desired_rotation_matrix[:, 2]
        desired_theta = np.arccos(desired_z_b_0[2])
        if desired_theta > self.rot_limits or real_theta > self.rot_limits:
            rot_trigger = True
        if rot_trigger == True:
            self.get_logger().info("Rotation Securtity triggered, attempting return to horizontal orientation", throttle_duration_sec = 1)
            if self.selected_controller !=2:
                self.controller_selection(2, init = True, reset = False)
            self.reset_trajectory_client.call_async(Empty.Request())
            self.controller.desired_pose.position = np.copy(self.real_pose.position)
            self.controller.desired_pose.rotation = Quaternion()

    def controller_execute(self):
        self.sec = self.odom.header.stamp.sec
        self.nanosec = self.odom.header.stamp.nanosec
        self.time_prev = self.time
        self.time = self.sec + self.nanosec*10**(-9)
        self.step_size_old = self.step_size
        self.step_size = self.time - self.time_prev
        if self.step_size < 0:
            self.step_size = self.step_size_old
        self.real_pose = self.odom2tilthex_pose(self.odom)
        if (self.controller is not None):
            # Do control
            if self.controller.type is Tilthex_Controller_Type.TEST:
                control = self.controller.do_control()
            else: 
                if self.taking_off_flag:
                    self.taking_off()
                if self.landing_flag:
                    self.landing()
                if self.flying_flag:
                    self.flying()
                if self.experiment_started:
                    if not self.landing_flag:
                        self.rotation_security()
                        self.virtual_safety_net()
                    if self.controller.type == Tilthex_Controller_Type.GEOMETRIC:
                        control = self.controller.do_control(self.real_pose, self.step_size, self.anti_windup_trans_switch, self.anti_windup_rot_switch)
                    if self.controller.type == Tilthex_Controller_Type.VELOCITY:
                        control = self.controller.do_control(self.real_pose, self.step_size)
                    self.direct_motor_control_transfer(control)

                    # Debug
                    self.classic_debug()
                    self.real_pose_debug()
                    self.desired_pose_debug()
                    self.take_off_pose_debug()


def main(args=None):
    rclpy.init(args=args)
    node = TilthexControlCenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down control center")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
