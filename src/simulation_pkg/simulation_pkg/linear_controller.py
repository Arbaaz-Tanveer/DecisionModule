#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

class MultiRobotController(Node):
    def __init__(self):
        super().__init__('multi_robot_controller')
        
        # Controller parameters
        self.MAX_LINEAR_VEL = 3  # m/s
        self.MAX_ANGULAR_VEL = 2.0  # rad/s
        self.MAX_LINEAR_ACC = 2   # m/s^2
        self.MAX_ANGULAR_ACC = 1  # rad/s^2
        self.POSITION_THRESHOLD = 0.05  # meters
        self.ANGLE_THRESHOLD = 0.05     # radians
        
        self.num_robots = 5
        self.current_poses = [None] * self.num_robots
        self.target_poses = [None] * self.num_robots
        self.prev_linear_vel = [0.0] * self.num_robots
        self.prev_angular_vel = [0.0] * self.num_robots
        self.reached_target = [False] * self.num_robots  # Track if each robot has reached its target
        self.target_update_time = [0.0] * self.num_robots  # Track when each target was last updated
        self.dt = 0.05  # 10Hz control loop
        
        # Initialize publishers and subscribers for each robot
        self.cmd_vel_pubs = []
        self.pose_subs = []
        self.target_subs = []
        
        for i in range(self.num_robots):
            # Create publishers
            pub = self.create_publisher(
                Twist,
                f'o{i+1}/cmd_vel',
                10
            )
            self.cmd_vel_pubs.append(pub)
            
            # Create subscribers for current pose
            pose_sub = self.create_subscription(
                Float32MultiArray,
                f'o{i+1}_data',
                lambda msg, robot_id=i: self.pose_callback(msg, robot_id),
                10
            )
            self.pose_subs.append(pose_sub)
            
            # Create subscribers for target pose
            target_sub = self.create_subscription(
                Float32MultiArray,
                f'/o{i+1}/decision_target_data',
                lambda msg, robot_id=i: self.target_callback(msg, robot_id),
                10
            )
            self.target_subs.append(target_sub)
        
        # Create timer for control loop
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info('MultiRobotController initialized')
    
    def pose_callback(self, msg, robot_id):
        try:
            if len(msg.data) >= 3:
                self.current_poses[robot_id] = [msg.data[0], msg.data[1], msg.data[2]]
                self.get_logger().debug(f'Robot {robot_id} at position: {self.current_poses[robot_id]}')
            else:
                self.get_logger().warning(f'Received incomplete pose data for robot {robot_id}')
        except Exception as e:
            self.get_logger().error(f'Error in pose_callback: {str(e)}')
    
    def target_callback(self, msg, robot_id):
        try:
            if len(msg.data) >= 3:
                new_target = [msg.data[0], msg.data[1], msg.data[2]]
                
                # Check if this is a new target
                if (self.target_poses[robot_id] is None or 
                    not np.allclose(new_target, self.target_poses[robot_id], atol=1e-3)):
                    self.target_poses[robot_id] = new_target
                    self.reached_target[robot_id] = False  # Reset reached_target flag
                    self.target_update_time[robot_id] = self.get_clock().now().nanoseconds / 1e9
                    self.get_logger().info(f'New target for robot {robot_id}: {new_target}')
        except Exception as e:
            self.get_logger().error(f'Error in target_callback: {str(e)}')
    
    def calculate_control(self, current, target, prev_linear_vel, prev_angular_vel):
        """
        Computes global velocity commands given current and target poses.
        The output velocities (vx, vy) are in the global frame.
        When the robot is far from the target position, no angular velocity is commanded.
        Once the robot nears the target, orientation is corrected.
        """
        if current is None or target is None:
            return 0.0, 0.0, 0.0

        try:
            # Unpack the poses.
            x, y, theta = current            # current: global x, y, and current orientation (theta)
            target_x, target_y, target_theta = target  # target: global x, y, and desired final orientation

            # Compute the positional error.
            dx = target_x - x
            dy = target_y - y
            distance = np.sqrt(dx**2 + dy**2)

            # Use a proportional controller on the distance.
            k_v = 0.5  # linear gain (tune as needed)
            desired_speed = min(k_v * distance, self.MAX_LINEAR_VEL)

            # Determine the global direction to the target.
            global_direction = np.arctan2(dy, dx)
            # Compute the global linear velocities.
            vx = desired_speed * np.cos(global_direction)
            vy = desired_speed * np.sin(global_direction)

        
            orientation_error = self.normalize_angle(target_theta - theta)
            desired_omega = np.clip(1.0 * orientation_error, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

            # Apply acceleration limits.
            current_speed = np.sqrt(vx**2 + vy**2)
            limited_speed = self.apply_acceleration_limit(prev_linear_vel, current_speed, self.MAX_LINEAR_ACC, self.dt)
            if current_speed > 0:
                vx = limited_speed * np.cos(global_direction)
                vy = limited_speed * np.sin(global_direction)
            else:
                vx, vy = 0.0, 0.0

            limited_omega = self.apply_acceleration_limit(prev_angular_vel, desired_omega, self.MAX_ANGULAR_ACC, self.dt)

            return vx, vy, limited_omega

        except Exception as e:
            self.get_logger().error(f'Error in calculate_control: {str(e)}')
            return 0.0, 0.0, 0.0


    
    def apply_acceleration_limit(self, current_vel, desired_vel, max_acc, dt):
        max_vel_change = max_acc * dt
        return np.clip(
            desired_vel,
            current_vel - max_vel_change,
            current_vel + max_vel_change
        )
    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        try:
            for i in range(self.num_robots):
                if self.current_poses[i] is None or self.target_poses[i] is None:
                    continue

                # Calculate control inputs.
                vx, vy, omega = self.calculate_control(
                    self.current_poses[i],
                    self.target_poses[i],
                    self.prev_linear_vel[i],
                    self.prev_angular_vel[i]
                )

                # Check if the target is reached.
                x, y, theta = self.current_poses[i]
                target_x, target_y, target_theta = self.target_poses[i]
                pos_error = np.sqrt((target_x - x)**2 + (target_y - y)**2)
                angle_error = abs(self.normalize_angle(target_theta - theta))
                if pos_error < self.POSITION_THRESHOLD and angle_error < self.ANGLE_THRESHOLD:
                    self.reached_target[i] = True
                    self.get_logger().info(f'Robot {i} reached target')
                else:
                    self.reached_target[i] = False

                # Update previous velocities.
                self.prev_linear_vel[i] = np.sqrt(vx**2 + vy**2)
                self.prev_angular_vel[i] = omega

                # Publish command velocities.
                cmd_vel = Twist()
                cmd_vel.linear.x = vx  # Global linear x velocity.
                cmd_vel.linear.y = vy  # Global linear y velocity.
                cmd_vel.angular.z = omega  # Angular velocity command.
                self.cmd_vel_pubs[i].publish(cmd_vel)

        except Exception as e:
            self.get_logger().error(f'Error in control_loop: {str(e)}')



def main(args=None):
    rclpy.init(args=args)
    controller = MultiRobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()