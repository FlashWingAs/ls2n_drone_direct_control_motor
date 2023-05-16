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
)
from ls2n_drone_bridge.drone_bridge import (
    Status
)

class ControlCenter(Node):
    def __init__(self):
        super().__init__("direct_motor_control")
        parameters = [
            [
                "mass",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Drone mass"],
            [
                "take_off_height",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Height for automatic take off",
            ],
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
                "kp_trans_geom",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric translation controller proportionnal gain"
            ],
            [
                "ki_trans_geom",
                0.1,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric translation controller integral gain"
            ],
            [
                "kd_trans_geom",
                0.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric translation controller derivative gain"
            ],
            [
                "kp_rot_geom",
                0.5,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric rotational controller proportionnal gain"
            ],
            [
                "ki_rot_geom",
                0.05,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric rotational controller integral gain"
            ],
            [
                "kd_rot_geom",
                0.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric rotational controller derivative gain"
            ],
            [
                "yaw_p_multiplier",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric yaw controller derivative multiplier"
            ],
            [
                "yaw_i_multiplier",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric yaw controller derivative multiplier"
            ],
            [
                "yaw_d_multiplier",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Geometric yaw controller derivative multiplier"
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

        self.get_logger().info("Starting control center node")

        self.create_service(Empty, "SpinMotors", self.spin_motors)
        self.create_service(Empty, "StartExperiment", self.start_experiment)
        self.create_service(Empty, "StopExperiment", self.stop_experiment)
        self.drone_request_client = self.create_client(
            DroneRequest,
            "Request",
        )
        self.experiment_started = False
        
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

        self.position_tracking_publisher = self.create_publisher(
            CustomDebug,
            "Posetracking",
            qos_profile_sensor_data
        )

        self.custom_pose_publisher = self.create_publisher(
            CustomPoseDebug,
            "Pose_translation",
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
    odom = FullState()
    real_pose = Custom_Pose()
    desired_pose = Custom_Pose()
    desired_pose.position = np.array([0.0, 0.0, 1.5]) #fixed set point of 1.5m altitude
    respond2trajectory = False
    controller = None
    sec = 0
    nanosec = 0.0
    time_prev = 0.0
    time = 0.0
    step_size = 0.0
    step_size_old = 0.0
    geometric_contoller_parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  
    # Callbacks

    def parameters_callback(self, params):
        pid_change = False
        desired_pose_change = False
        for param in params:
            if param.name == "kp_trans_geom":
                self.kp_trans_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom, self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "ki_trans_geom":
                self.ki_trans_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom, self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "kd_trans_geom":
                self.kd_trans_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom,
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "kp_rot_geom":
                self.kp_rot_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom, self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "ki_rot_geom":
                self.ki_rot_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom, self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "kd_rot_geom":
                self.kd_rot_geom = param.value
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom,
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "yaw_p_multiplier":
                self.yaw_p_multiplier = param.name
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier, self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "yaw_i_multiplier":
                self.yaw_p_multiplier = param.name
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier, self.yaw_d_multiplier()]
                pid_change = True
            if param.name == "yaw_d_multiplier":
                self.yaw_p_multiplier = param.name
                self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                           self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier]
                pid_change = True
            if param.name == "d_position":
                self.d_postion = param.value
                self.desired_pose.position = np.fromstring(param.value, sep = ',')
                desired_pose_change = True
            if param.name == "d_rot_euler":
                self.d_rot_euler = param.value
                self.desired_pose.rotation = Quaternion(np.roll(R.from_euler('XYZ', np.fromstring(param.value, sep = ','), degrees = True).as_quat()))
                desired_pose_change = True
        if pid_change:
            self.controller.update_pid(self.geometric_contoller_parameters)
            self.get_logger().info("PID parameters changed succesfully")
        if desired_pose_change:
            self.controller.update_desired_pose(self.desired_pose)
            self.get_logger().info("Desired pose changed succesfully")
        return SetParametersResult(successful=True)

    def status_update_callback(self, rcvd_msg):
        self.status.status = rcvd_msg.status
        if (self.status.status == DroneStatus.FLYING and not self.experiment_started):
            self.experiment_started = True
        if (self.status.status == DroneStatus.IDLE and self.experiment_started):
            self.experiment_started = False

    def odom_update_callback(self, odom):
        self.odom = odom
        self.controller_go_brrr()
        
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
            if abs < 0.0:
                abs = 0
            to_send[i] = abs
        if max > 1:
            to_send = to_send / max
        to_send_L = to_send.tolist()
        msg = MotorControlSetPoint()
        msg.motor_velocity = to_send_L
        self.direct_motor_control_publisher.publish(msg)

    def position_tracking(self, CurrentPose, Go2Pose, V_vec):
        msg = CustomDebug()
        msg.step_size = float(self.step_size)
        msg.v_vec = V_vec
        self.position_tracking_publisher.publish(msg)

    def real_pose_debug(self, CurrentPose):
        msg = CustomPoseDebug()
        msg.position = CurrentPose.position.tolist()
        msg.velocity = CurrentPose.velocity.tolist()
        msg.acceleration = CurrentPose.acceleration.tolist()
        # msg.rotation = np.concatenate((CurrentPose.rotation.scalar,CurrentPose.rotation.vector)).tolist()
        msg.rot_velocity = CurrentPose.rot_velocity.tolist()
        msg.rot_acceleration = CurrentPose.rot_acceleration.tolist()
        self.custom_pose_publisher.publish(msg)

    # Services

    def spin_motors(self, request, response):
        if not self.experiment_started:
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
                self.contoller_selection(self.select_controller())
                if self.controller.type == Custom_Controller_Type.GEOMETRIC:
                    if self.config_switch():
                        self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                                self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom(),
                                                                self.yaw_p_multiplier(), self.yaw_i_multiplier(), self.yaw_d_multiplier()]
                        self.controller.init_controller(do_trajectory = self.respond2trajectory, take_off_pose = self.desired_pose, parameters = self.geometric_contoller_parameters)
                    else:
                        self.controller.init_controller(do_trajectory = self.respond2trajectory, take_off_pose = self.desired_pose)
                if self.controller.type == Custom_Controller_Type.TEST:
                    pass
                request_out = DroneRequest.Request()
                request_out.request = DroneRequest.Request.MOTOR_CONTROL
                self.drone_request_client.call_async(request_out)
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

    def odom2state(self, odometry):
        real_state = Custom_Pose()
        real_state.position[0] = odometry.pose.pose.position.x
        real_state.position[1] = odometry.pose.pose.position.y
        real_state.position[2] = odometry.pose.pose.position.z
        real_state.velocity[0] = odometry.twist.twist.linear.x
        real_state.velocity[1] = odometry.twist.twist.linear.y
        real_state.velocity[2] = odometry.twist.twist.linear.z
        real_state.rotation = Quaternion(odometry.pose.pose.orientation.w, odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z)
        real_state.rot_velocity[0] = odometry.twist.twist.angular.x
        real_state.rot_velocity[1] = odometry.twist.twist.angular.y
        real_state.rot_velocity[2] = odometry.twist.twist.angular.z
        return real_state

    def thrust2abs(self, thrust: float) -> float:
        abs = thrust/37.0
        return abs

    # Main loop

    def main_loop(self):
        pass

    # Special actions

    def controller_go_brrr(self):
        self.sec = self.odom.header.stamp.sec
        self.nanosec = self.odom.header.stamp.nanosec
        self.time_prev = self.time
        self.time = self.sec + self.nanosec*10**(-9)
        self.step_size_old = self.step_size
        self.step_size = self.time - self.time_prev
        if self.step_size < 0:
            self.step_size = self.step_size_old

        self.real_pose = self.odom2state(self.odom)
        if (self.controller is not None) and self.experiment_started:
            # Do control
            if self.controller.type is Custom_Controller_Type.TEST:
                control = self.controller.do_control()
            else: 
                control = self.controller.do_control(self.real_pose, self.step_size)
            self.direct_motor_control_transfer(control)

            # Debug
            desired_pose = self.controller.debug_controller_desired_pose()
            V_vec = self.controller.debug_v()
            self.position_tracking(self.real_pose, desired_pose, V_vec)
            self.real_pose_debug(self.real_pose)


def main(args=None):
    rclpy.init(args=args)
    node = ControlCenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down control center")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
