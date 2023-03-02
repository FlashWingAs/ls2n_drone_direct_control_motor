import importlib
import numpy as np
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
from ls2n_drone_bridge.drone_bridge import (
    Status
)

class ControlCenter(Node):
    def __init__(self):
        super().__init__("direct_motor_control")
        parameters = [
            ["mass", 1.0, ParameterType.PARAMETER_DOUBLE, "Drone mass"],
            [
                "take_off_height",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Height for automatic take off",
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

        self.status_timer = self.create_timer(0.01, self.status_publisher)

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
    odometry = FullState()
    controller = None
  
    # Callbacks

    def status_update_callback(self,rcvd_msg):
        self.status.status = rcvd_msg.status
        if (self.status.status == DroneStatus.FLYING and not self.experiment_started):
            self.experiment_started = True
        if (self.status.status == DroneStatus.IDLE and self.experiment_started):
            self.experiment_started = False

    def odom_update_callback(self, msg):
        self.odometry = msg
        if (self.controller is not None) and self.experiment_started:
            control = self.controller.do_control()
            self.direct_motor_control_publisher(control)
        

    # Publishers

    def status_publisher(self):
        msg = DroneStatus()
        msg.status = self.status.status
        self.status_publisher.publish(msg)

    def direct_motor_control_publisher(self, motors_set_points):
        if len(motors_set_points)<12:
            for i in range(12 - len(motors_set_points)):
                motors_set_points.append(0.0)
        msg = MotorControlSetPoint()
        msg.data = motors_set_points
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
            self.controller = Geometric_Controller()
            self.status.status = DroneStatus.FLYING
            self.experiment_started = True
        else:
            self.get_logger().info(
                "Experiment already started. Stop experiment to reset."
            )
        return response

    def main_loop(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = ControlCenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down drone bridge")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
