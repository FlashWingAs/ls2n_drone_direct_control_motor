import importlib
import numpy as np
from scipy.spatial.transform import Rotation as R
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
)
from ls2n_interfaces.srv import DroneRequest, OnboardApp
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.parameter import Parameter

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

        self.status_publisher = self.create_publisher(
            DroneStatus,
            "Status",
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

    status = Status()
    odom = FullState()
    real_pose = Custom_Pose()
    respond2trajectory = False
    controller = None
    sec = 0
    nanosec = 0.0
    time_prev = 0.0
    time = 0.0
    step_size = 0.0
    step_size_old = 0.0
    geometric_contoller_parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  
    # Callbacks

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

    def status_update(self):
        msg = DroneStatus()
        msg.status = self.status.status
        self.status_publisher.publish(msg)

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

    def position_tracking(self, CurrentPose, Go2Pose):
        msg = CustomDebug()
        msg.current_position = CurrentPose.position.tolist()
        msg.current_rotation_vec = CurrentPose.rotation.tolist()
        msg.goal_position = Go2Pose.position.tolist()
        msg.goal_rotation_vec = Go2Pose.rotation.tolist()
        msg.error_position = (Go2Pose.position - CurrentPose.position).tolist()
        msg.error_rotation_vec = (Go2Pose.rotation - CurrentPose.rotation).tolist()
        msg.bx_in_r = CurrentPose.rotation_matrix[:,0].tolist()
        msg.by_in_r = CurrentPose.rotation_matrix[:,1].tolist()
        msg.bz_in_r = CurrentPose.rotation_matrix[:,2].tolist()

        msg.step_size = float(self.step_size)
        self.position_tracking_publisher.publish(msg)

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
                    self.geometric_contoller_parameters = [self.kp_trans_geom(), self.ki_trans_geom(), self.kd_trans_geom(),
                                                           self.kp_rot_geom(), self.ki_rot_geom(), self.kd_rot_geom()]
                if self.config_switch():
                    self.controller.init_controller(parameters = self.geometric_contoller_parameters)
                if self.controller.type == Custom_Controller_Type.TEST:
                    pass
                self.status.status = DroneStatus.FLYING
                self.status_update()
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
            self.status.status = DroneStatus.IDLE
            self.status_update()
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
        real_state.rotation[0] = odometry.pose.pose.orientation.x
        real_state.rotation[1] = odometry.pose.pose.orientation.y
        real_state.rotation[2] = odometry.pose.pose.orientation.z
        real_state.rot_velocity[0] = odometry.twist.twist.angular.x
        real_state.rot_velocity[1] = odometry.twist.twist.angular.y
        real_state.rot_velocity[2] = odometry.twist.twist.angular.z
        real_state.rotation_matrix = R.from_rotvec(real_state.rotation).as_matrix()
        real_state.rotation_matrix_derivative = R.from_rotvec(real_state.rot_velocity).as_matrix()
        return real_state
    
    def rad2abs(self, rad):
        abs = (0.002*rad-66.38)/222.0
        return abs

    def thrust2abs(self, thrust: float) -> float:
        abs = thrust/222.0
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
                control = self.controller.do_control(self.real_pose, self.step_size, self.respond2trajectory)
            self.direct_motor_control_transfer(control)

            # Debug
            desired_pose = self.controller.debug_controller_desired_pose()
            self.position_tracking(self.real_pose, desired_pose)


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
