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
                "controller",
                1,
                ParameterType.PARAMETER_INTEGER,
                "Custom controller selection"
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

        self.create_service(
            Empty,
            "SpinMotors",
            self.spin_motors,
        )
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
        
        # Set points
        self.direct_motor_control_publisher = self.create_publisher(
            MotorControlSetPoint,
            "MotorControlSetPoint",
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
    desired_pose = Custom_Pose()
    desired_pose.position = np.array([0.0, 0.0, 2.0])
    controller = None
    sec = 0
    nanosec = 0.0
    time_prev = 0.0
    time = 0.0
    step_size = 0.0
  
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
        
        

    # Publishers

    def status_update(self):
        msg = DroneStatus()
        msg.status = self.status.status
        self.status_publisher.publish(msg)

    def direct_motor_control_transfer(self, motors_set_points):
        to_send = np.zeros(12)
        max = 0
        for i in range(len(motors_set_points)):
            abs = self.rad2abs(motors_set_points[i])
            if abs > max:
                max = abs
            to_send[i] = abs
        if max > 1:
            to_send = to_send / max
        to_send_L = to_send.tolist()
        msg = MotorControlSetPoint()
        msg.motor_velocity = to_send_L
        self.direct_motor_control_publisher.publish(msg)

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
                self.controller = Geometric_Controller(self)
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
        real_state.orientation_matrix = R.from_euler('ZYX', real_state.rotation).as_matrix()
        return real_state
    
    def rad2abs(self, rad):
        abs = rad
        return abs

    # Main loop

    def main_loop(self):
        pass

    # Special actions

    def controller_go_brrr(self):
        self.sec = self.odom.header.stamp.sec
        self.nanosec = self.odom.header.stamp.nanosec
        self.time_prev = self.time
        self.time = self.sec + self.nanosec
        self.step_size = self.time - self.time_prev

        self.real_pose = self.odom2state(self.odom)
        if (self.controller is not None) and self.experiment_started:
            if self.controller.type is Custom_Controller_Type.TEST:
                control = self.controller.do_control()
            else: 
                control = self.controller.do_control(self.real_pose, self.desired_pose, self.step_size)
            self.direct_motor_control_transfer(control)


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
