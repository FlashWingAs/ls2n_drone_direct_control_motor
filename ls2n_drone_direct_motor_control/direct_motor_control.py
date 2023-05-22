import importlib
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import os
import subprocess
import signal
import rclpy
from geometry_msgs.msg import Vector3Stamped
from ls2n_interfaces.msg import (
    AttitudeThrustSetPoint,
    DroneStatus,
    RatesThrustSetPoint,
    KeepAlive,
    MotorControlSetPoint,
    CustomDebug,
    CustomPoseDebug,
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
from ls2n_drone_direct_motor_control.custom_controllers import (
    Test_Controller,
    Geometric_Controller
)
from ls2n_drone_direct_motor_control.custom_common import (
    Custom_Controller_Type,
    Custom_Pose,
    Custom_PID_Param,
)
from ls2n_drone_bridge.drone_bridge import (
    Status
)

class CustomControlCenter(Node):
    def __init__(self):
        super().__init__("direct_motor_control")
        parameters = [
            [
                "mass",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Drone mass"],
            [
                "max_thrust",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Total drone max thrust"],
            [
                "select_controller",
                2,
                ParameterType.PARAMETER_INTEGER,
                "Custom controller selection"
            ],
            [
                "config_switch",
                False,
                ParameterType.PARAMETER_BOOL,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_trans_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_trans_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_trans_d",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_rot_p",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_rot_i",
                "0.0, 0.0, 0.0",
                ParameterType.PARAMETER_STRING,
                "Switch if PID config is default or handpicked"
            ],
            [
                "pid_rot_d",
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
        
        self.custom_pid_switch = False                   # True if using default pid parameters set in the controller
        i = 0
        if self.pid_trans_p != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
        if self.pid_trans_i != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
        if self.pid_trans_d != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
        if self.pid_rot_p != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
        if self.pid_rot_i != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
        if self.pid_rot_d != "0.0, 0.0, 0.0":
            self.custom_pid_switch = True
          

        self.get_logger().info("Starting control center node")

        # Service creation
        self.create_service(Empty, "SpinMotors", self.spin_motors)
        self.create_service(Empty, "StartExperiment", self.start_experiment)
        self.create_service(Empty, "StopExperiment", self.stop_experiment)
        self.drone_request_client = self.create_client(
            DroneRequest,
            "Request",
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
            CustomDebug,
            "classic_debug",
            qos_profile_sensor_data
        )

        self.real_pose_debug_publisher = self.create_publisher(
            CustomPoseDebug,
            "real_pose",
            qos_profile_sensor_data
        )

        self.desired_pose_debug_publisher = self.create_publisher(
            CustomPoseDebug,
            "desired_pose",
            qos_profile_sensor_data
        )

        self.take_off_pose_debug_publisher = self.create_publisher(
            CustomPoseDebug,
            "take_off_pose",
            qos_profile_sensor_data
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
        
        self.add_on_set_parameters_callback(self.parameters_callback)

    status = Status()
    experiment_started = False
    odom = Odometry()
    real_pose = Custom_Pose()
    respond2trajectory = False
    take_off_pose = Custom_Pose()
    controller = None
    sec = 0
    nanosec = 0.0
    time = 0.0
    time_prev = 0.0
    step_size = 0.0
    step_size_old = 0.0
  
    # Callbacks

    def parameters_callback(self, params):
        for param in params:
            if self.controller.type == Custom_Controller_Type.GEOMETRIC:
                if param.name == "pid_trans_p":
                    self.controller.PID.trans_p = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "pid_trans_i":
                    self.controller.PID.trans_i = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "pid_trans_d":
                    self.controller.PID.trans_d = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "pid_rot_p":
                    self.controller.PID.rot_p = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "pid_rot_i":
                    self.controller.PID.rot_i = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "pid_rot_d":
                    self.controller.PID.rot_d = np.fromstring(param.value, sep = ",")
                    self.controller.PID.update()
                if param.name == "d_position":
                    self.controller.desired_pose.position = np.fromstring(param.value, sep = ',')
                if param.name == "d_rot_euler":
                    self.controller.desired_pose.rotation = Quaternion(np.roll(R.from_euler('ZYX', np.fromstring(param.value, sep = ','), degrees = True).as_quat(), 1))
                
                self.get_logger().info("Parameter succesfully updated")
            else:
                self.get_logger().info("Controller not initialised, cannot update parameters")
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
            if self.controller.type == Custom_Controller_Type.GEOMETRIC:
                if self.status.status == DroneStatus.FLYING and self.experiment_started:
                    if self.respond2trajectory: 
                        self.controller.update_setpoints(msg)

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
        msg = CustomDebug()
        msg.step_size = float(self.step_size)
        msg.v_b = self.controller.v_B
        msg.v_p = self.controller.Vp
        msg.v_r = self.controller.Vr
        msg.current_position = self.real_pose.position.tolist()
        msg.current_rotation = np.concatenate((np.array([self.real_pose.rotation.scalar]), self.real_pose.rotation.vector)).tolist()
        msg.goal_position = self.controller.desired_pose.position.tolist()
        msg.goal_rotation = np.concatenate((np.array([self.controller.desired_pose.rotation.scalar]), self.controller.desired_pose.rotation.vector)).tolist()
        msg.error_position = self.controller.e_pos.tolist()
        msg._error_rotation = self.controller.e_ang.tolist()
        self.classic_debug_publisher.publish(msg)

    def real_pose_debug(self):
        msg = CustomPoseDebug()
        msg.position = self.real_pose.position.tolist()
        msg.velocity = self.real_pose.velocity.tolist()
        msg.acceleration = self.real_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.real_pose.rotation.scalar]), self.real_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.real_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.real_pose.rot_acceleration.tolist()
        self.real_pose_debug_publisher.publish(msg)

    def desired_pose_debug(self):
        msg = CustomPoseDebug()
        msg.position = self.controller.desired_pose.position.tolist()
        msg.velocity = self.controller.desired_pose.velocity.tolist()
        msg.acceleration = self.controller.desired_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.controller.desired_pose.rotation.scalar]), self.controller.desired_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.controller.desired_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.controller.desired_pose.rot_acceleration.tolist()
        self.desired_pose_debug_publisher.publish(msg)

    def take_off_pose_debug(self):
        msg = CustomPoseDebug()
        msg.position = self.take_off_pose.position.tolist()
        msg.velocity = self.take_off_pose.velocity.tolist()
        msg.acceleration = self.take_off_pose.acceleration.tolist()
        msg.rotation = np.concatenate((np.array([self.take_off_pose.rotation.scalar]), self.take_off_pose.rotation.vector)).tolist()
        msg.rot_velocity = self.take_off_pose.rot_velocity.tolist()
        msg.rot_acceleration = self.take_off_pose.rot_acceleration.tolist()
        self.take_off_pose_debug_publisher.publish(msg)

    # Services

    def spin_motors(self, request, response):
        if not self.experiment_started:
            self.contoller_selection(self.select_controller())
            if self.controller.type == Custom_Controller_Type.GEOMETRIC:
                self.controller.m_tot = self.mass()
                if self.custom_pid_switch:
                    self.controller.PID = Custom_PID_Param(np.fromstring(self.pid_trans_p(), sep = ','),
                                                           np.fromstring(self.pid_trans_i(), sep = ','),
                                                           np.fromstring(self.pid_trans_d(), sep = ','),
                                                           np.fromstring(self.pid_rot_p(), sep = ','),
                                                           np.fromstring(self.pid_rot_i(), sep = ','),
                                                           np.fromstring(self.pid_rot_d(), sep = ','))
                    self.get_logger().info("Initilised with custom PID parameters")
                else:
                    self.get_logger().info("Initilised with default PID parameters")
                if not self.respond2trajectory:
                        self.take_off_pose.position = np.fromstring(self.d_position(), sep=",")
                        self.controller.desired_pose = self.take_off_pose
            if self.controller.type == Custom_Controller_Type.TEST:
                pass
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
                request_out.request = DroneRequest.Request.DIRECT_MOTOR_CONTROL
                self.drone_request_client.call_async(request_out)
                self.experiment_started = True
                self.get_logger().info("Experiment starting")
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
            self.get_logger().info("Stopping drone")
            self.controller = None
            request_out = DroneRequest.Request()
            request_out.request = DroneRequest.Request.DISARM
            self.drone_request_client.call_async(request_out)
            self.experiment_started = False
        else:
            self.get_logger().info(
                "Experiment already stopped."
            )
        return response
    
    # State machine

    def contoller_selection(self, selection):
        match selection:
            case 1:
                self.controller = Test_Controller(self)
                self.get_logger().info("Experiment starting with TEST controller")
            case 2:
                self.controller = Geometric_Controller(self)
                self.get_logger().info("Experiment starting with GEOMETRIC controller")

    # Translations

    def odom2custom_pose(self, odometry):
        custom = Custom_Pose()
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
        abs = thrust/(self.max_thrust()/6.0)
        return abs

    # Main loop

    def main_loop(self):
        pass

    # Special actions

    def controller_execute(self):
        self.sec = self.odom.header.stamp.sec
        self.nanosec = self.odom.header.stamp.nanosec
        self.time_prev = self.time
        self.time = self.sec + self.nanosec*10**(-9)
        self.step_size_old = self.step_size
        self.step_size = self.time - self.time_prev
        if self.step_size < 0:
            self.step_size = self.step_size_old
        self.real_pose = self.odom2custom_pose(self.odom)
        if (self.controller is not None) and self.experiment_started:
            # Do control
            if self.controller.type is Custom_Controller_Type.TEST:
                control = self.controller.do_control()
            else: 
                control = self.controller.do_control(self.real_pose, self.step_size)
            self.direct_motor_control_transfer(control)

            # Debug
            self.classic_debug()
            self.real_pose_debug()
            self.desired_pose_debug()
            self.take_off_pose_debug()


def main(args=None):
    rclpy.init(args=args)
    node = CustomControlCenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down control center")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
