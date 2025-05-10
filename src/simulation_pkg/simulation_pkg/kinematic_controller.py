#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')
        
        # Publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/o1/cmd_vel', 10)
        
        # Subscribers for current state and waypoints
        self.create_subscription(Float32MultiArray, 'o1_data', self.current_state_callback, 10)
        self.create_subscription(Float32MultiArray, '/o1/target_pos', self.target_state_callback, 10)
        
        # Current robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_omega = 0.0
        
        # Previous waypoint (point A)
        self.prev_waypoint_x = 0.0
        self.prev_waypoint_y = 0.0
        
        # Current waypoint (point B)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        self.target_vx = 0.1
        self.target_vy = 0.0
        self.target_omega = 0.0
        self.idx = 0
        self.path_length = 0
        
        # Controller parameters (tunable)
        self.max_linear_accel = 2      # m/s²
        self.max_linear_velocity = 3  # m/s
        self.max_angular_accel = 0.5   # rad/s²
        self.position_tolerance = 0.01 # m
        self.final_position_tolerance = 0.005  # tighter tolerance for final damping
        self.orientation_tolerance = 0.05 # rad
        self.min_velocity = 0.05       # m/s (avoid division by near-zero)
        self.min_time = 0.1            # s (minimum time estimate)
        self.epsilon = 1e-6            # small number for comparisons
        self.min_parallel_velocity = 0.2  # Minimum velocity to maintain in parallel direction
        self.damping_factor = 0.8      # Velocity damping factor for final position approach
        
        # Time tracking for dt calculation
        self.last_time = time.time()
        
        # Store last commanded velocities
        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0
        self.last_cmd_omega = 0.0
        
        # For debugging
        self.debug_counter = 0
        
        # For trajectory calculations
        self.unit_parallel = (1.0, 0.0)  # Default direction
        self.unit_perp = (0.0, 1.0)      # Default perpendicular
        
        # Flag to track final approach
        self.final_approach = False
        
        # Set logging level
        self.debug_mode = False
        
        self.get_logger().info('Trajectory Controller initialized')

    def target_state_callback(self, msg):
        """Process new waypoint information"""
        if len(msg.data) < 6:
            self.get_logger().error('Received target message with insufficient data')
            return
            
        # Extract target state (point B)
        self.target_x = msg.data[0]
        self.target_y = msg.data[1]
        self.target_theta = msg.data[2]
        self.target_vx = msg.data[3]
        self.target_vy = msg.data[4]
        self.target_omega = msg.data[5]
        self.idx = msg.data[7]
        self.path_length = msg.data[8]

        # Extract previous waypoint (point A) if available
        if len(msg.data) > 10:
            self.prev_waypoint_x = msg.data[9]
            self.prev_waypoint_y = msg.data[10]
        else:
            # If no previous waypoint provided, use current position as default
            self.prev_waypoint_x = self.current_x
            self.prev_waypoint_y = self.current_y
            
        # Reset final approach flag for new waypoint
        self.final_approach = False
            
        self.get_logger().info(f'New target: idx={self.idx}/{self.path_length}, '
                              f'pos=({self.target_x:.2f}, {self.target_y:.2f}), '
                              f'theta={self.target_theta:.2f}')

    def current_state_callback(self, msg):
        """Process updated robot state and compute control"""
        if len(msg.data) < 3:
            self.get_logger().error('Received state message with insufficient data')
            return
            
        # Calculate dt since last update
        current_time = time.time()
        dt = (current_time - self.last_time)  
        self.last_time = current_time
        
        # Ensure dt is reasonable (handle initialization and any time jumps)
        if dt <= 0.0 or dt > 1.0:
            dt = 0.02  # Fallback to 50 Hz nominal
            
        # Update current state
        self.current_x = msg.data[0]
        self.current_y = msg.data[1]
        self.current_theta = msg.data[2]
        
        # Update velocities if available in message
        if len(msg.data) > 8:
            self.current_vx = msg.data[7]
            self.current_vy = msg.data[8]
        
        # Compute and publish control command
        cmd = self.compute_trajectory_control(dt)
        self.cmd_vel_pub.publish(cmd)
        
        # Store last commanded velocities
        self.last_cmd_vx = cmd.linear.x
        self.last_cmd_vy = cmd.linear.y
        self.last_cmd_omega = cmd.angular.z
        
        # Periodic detailed debug output
        self.debug_counter += 1
        if self.debug_counter % 5 == 0:  # Reduced frequency of logging
            self.get_logger().info(
                f'State: pos=({self.current_x:.2f}, {self.current_y:.2f}), '
                f'vel=({self.current_vx:.2f}, {self.current_vy:.2f}), '
                f'speed = {math.sqrt(self.current_vx**2 + self.current_vy**2)}, '
                f'cmd=({self.last_cmd_vx:.2f}, {self.last_cmd_vy:.2f}), '
                f'theta={self.current_theta:.2f}'
            )

    def compute_trajectory_control(self, dt):
        """Compute control commands using kinematic equations"""
        twist = Twist()
        
        # Calculate distance to target
        dx_to_target = self.target_x - self.current_x
        dy_to_target = self.target_y - self.current_y
        distance_to_target = math.sqrt(dx_to_target**2 + dy_to_target**2)
        
        # Calculate current velocity magnitude
        current_velocity_magnitude = math.sqrt(self.last_cmd_vx**2 + self.last_cmd_vy**2)
        
        # Check if we've reached the final position
        if distance_to_target < self.position_tolerance:
            # If we're at the target position, set velocities to zero and maintain target theta
            if current_velocity_magnitude < self.min_velocity:
                # If velocity is already very low, stop completely
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                
                # Control orientation to match target_theta
                delta_theta = self.normalize_angle(self.target_theta - self.current_theta)
                if abs(delta_theta) < self.orientation_tolerance:
                    twist.angular.z = 0.0
                    self.get_logger().info('Target position and orientation reached')
                else:
                    twist.angular.z = delta_theta * 2.0  # Proportional control
                    twist.angular.z = self.clamp(twist.angular.z, -1.0, 1.0)
            else:
                # Apply strong damping to smoothly stop the robot
                damping = 0.5  # Stronger damping for final stop
                twist.linear.x = self.last_cmd_vx * damping
                twist.linear.y = self.last_cmd_vy * damping
                
                # Control orientation to match target_theta
                delta_theta = self.normalize_angle(self.target_theta - self.current_theta)
                twist.angular.z = delta_theta * 2.0  # Proportional control
                twist.angular.z = self.clamp(twist.angular.z, -1.0, 1.0)
            
            return twist
        
        # Start applying extra damping when approaching final position
        if distance_to_target < 5 * self.position_tolerance and not self.final_approach:
            self.final_approach = True
            self.get_logger().info('Final approach initiated, applying velocity damping')
        
        # 2. Compute trajectory vectors and project current position
        self.compute_trajectory_vectors()
        
        # 3. Project current position and velocity onto trajectory
        parallel_dist, perp_dist, remaining_parallel_dist, traj_length = self.project_position_onto_trajectory()
        v_parallel, v_perp, target_v_parallel = self.project_velocity_onto_trajectory()
        
        # 4. Calculate control accelerations
        a_parallel, t_parallel = self.calculate_parallel_acceleration(
            v_parallel, target_v_parallel, remaining_parallel_dist, dt)
        
        a_perp = self.calculate_perpendicular_acceleration(
            perp_dist, v_perp, t_parallel*0.7, dt)
        
        # 5. Calculate new velocities based on computed accelerations
        new_vx, new_vy = self.calculate_velocity_commands(a_parallel, a_perp, dt)
        
        # Apply additional damping during final approach
        if self.final_approach:
            damping_factor = self.damping_factor * (1 - math.exp(-distance_to_target / self.position_tolerance))
            new_vx *= damping_factor
            new_vy *= damping_factor
            if self.debug_mode:
                self.get_logger().debug(f'Applying final approach damping: factor={damping_factor:.3f}')
        
        # 6. Set velocity commands
        twist.linear.x = new_vx
        twist.linear.y = new_vy
        
        # Calculate desired orientation to match target_theta (not trajectory alignment)
        delta_theta = self.normalize_angle(self.target_theta - self.current_theta)
        
        # Proportional control for orientation
        twist.angular.z = delta_theta * 2.0  # Proportional gain
        twist.angular.z = self.clamp(twist.angular.z, -1.0, 1.0)
        
        return twist

    def compute_trajectory_vectors(self):
        """Compute unit vectors along and perpendicular to the trajectory"""
        # Compute trajectory vector (from waypoint A to B)
        dx_traj = self.target_x - self.prev_waypoint_x
        dy_traj = self.target_y - self.prev_waypoint_y
        traj_length = math.sqrt(dx_traj**2 + dy_traj**2)
        
        # Handle case where trajectory length is effectively zero
        if traj_length < self.epsilon:
            # Direct line to target if no meaningful trajectory
            dx_to_target = self.target_x - self.current_x
            dy_to_target = self.target_y - self.current_y
            distance_to_target = math.sqrt(dx_to_target**2 + dy_to_target**2)
            
            if distance_to_target > self.epsilon:
                self.unit_parallel = (dx_to_target / distance_to_target, 
                                     dy_to_target / distance_to_target)
            else:
                self.unit_parallel = (1.0, 0.0)  # Default direction
        else:
            # Normal case - compute unit vector along trajectory
            self.unit_parallel = (dx_traj / traj_length, dy_traj / traj_length)
        
        # Compute perpendicular unit vector (rotate 90 degrees counter-clockwise)
        self.unit_perp = (-self.unit_parallel[1], self.unit_parallel[0])
        
    def project_position_onto_trajectory(self):
        """Project current position onto trajectory line and calculate distances"""
        # Vector from prev waypoint to current position
        dx_from_prev = self.current_x - self.prev_waypoint_x
        dy_from_prev = self.current_y - self.prev_waypoint_y
        
        # Compute parallel and perpendicular distances
        parallel_dist = dx_from_prev * self.unit_parallel[0] + dy_from_prev * self.unit_parallel[1]
        perp_dist = dx_from_prev * self.unit_perp[0] + dy_from_prev * self.unit_perp[1]
        
        # Calculate trajectory length
        dx_traj = self.target_x - self.prev_waypoint_x
        dy_traj = self.target_y - self.prev_waypoint_y
        traj_length = math.sqrt(dx_traj**2 + dy_traj**2)
        
        # Calculate distance along trajectory to target
        remaining_parallel_dist = traj_length - parallel_dist
        
        return parallel_dist, perp_dist, remaining_parallel_dist, traj_length
        
    def project_velocity_onto_trajectory(self):
        """Project velocities onto parallel and perpendicular directions"""
        # Project current velocities
        v_parallel = self.last_cmd_vx * self.unit_parallel[0] + self.last_cmd_vy * self.unit_parallel[1]
        v_perp = self.last_cmd_vx * self.unit_perp[0] + self.last_cmd_vy * self.unit_perp[1]
        
        # Target velocity projected onto trajectory
        target_v_parallel = self.target_vx * self.unit_parallel[0] + self.target_vy * self.unit_parallel[1]
        
        # Add progressive velocity reduction for final points in trajectory
        if (self.path_length - self.idx) == 4:
            target_v_parallel *= 0.7
        elif (self.path_length - self.idx) == 3:
            target_v_parallel *= 0.5
        elif (self.path_length - self.idx) == 2:
            target_v_parallel *= 0.3
        elif (self.path_length - self.idx) == 1:  # Added condition for last point
            target_v_parallel *= 0.1  # Significantly reduce velocity for final waypoint

        return v_parallel, v_perp, target_v_parallel
        
    def calculate_parallel_acceleration(self, v_parallel, target_v_parallel, remaining_parallel_dist, dt):
        """Calculate acceleration along the trajectory direction"""
        # First, ensure we have a minimum velocity in the parallel direction if not close to target
        t_parallel = self.min_time
        direction_to_target = 1 if remaining_parallel_dist > 0 else -1
        a_parallel = 0
 
        if abs(target_v_parallel) == 0:  # final stopping case
            stopping_time = abs(v_parallel / self.max_linear_accel)
            stopping_dist = 1.4 * 0.5 * abs(v_parallel) * stopping_time
            t_parallel = stopping_time
            
            if v_parallel * direction_to_target >= 0:  # moving towards the target
                if abs(v_parallel) < self.max_linear_velocity and stopping_dist < abs(remaining_parallel_dist):
                    a_parallel = direction_to_target * self.max_linear_accel
                    t_parallel += abs(remaining_parallel_dist) / self.max_linear_velocity  # conservative estimate
                elif stopping_dist > abs(remaining_parallel_dist):
                    a_parallel = -(v_parallel**2) / (2 * remaining_parallel_dist)
            else:  # moving away from target
                a_parallel = direction_to_target * self.max_linear_accel
        else:  # normal case
            if v_parallel >= 0:
                required_accel = (target_v_parallel**2 - v_parallel**2) / (2 * remaining_parallel_dist)
                a_parallel = required_accel
                t_parallel = abs(target_v_parallel - v_parallel) / a_parallel if abs(a_parallel) > self.epsilon else self.min_time
            else:
                a_parallel = self.max_linear_accel
                s = v_parallel * v_parallel / (2 * a_parallel)
                acc_after_stop = (target_v_parallel**2) / (2 * (remaining_parallel_dist + s))
                t_parallel = 2 * abs(v_parallel) / a_parallel + abs(target_v_parallel) / acc_after_stop
        
        # Limit parallel acceleration
        a_parallel = self.clamp(a_parallel, -self.max_linear_accel, self.max_linear_accel)
        
        if self.debug_mode:
            self.get_logger().debug(
                f'PARALLEL: dist={remaining_parallel_dist:.3f}, v={v_parallel:.3f}, '
                f'target_v={target_v_parallel:.3f}, a={a_parallel:.3f}'
            )
        
        return a_parallel, t_parallel
    
    def calculate_perpendicular_acceleration(self, perp_dist, v_perp, t_parallel, dt):
        """Calculate acceleration perpendicular to the trajectory"""
        t_parallel = max(0.3, t_parallel)
        
        # A small threshold for zero distance/velocity
        epsilon = self.epsilon
        sign_perp_dist = 1 if perp_dist >= 0 else -1
        stopping_dist = (v_perp**2) / (2 * self.max_linear_accel)

        # If we're very close to the trajectory, simply damp the velocity
        if abs(perp_dist) < epsilon:
            a_perp = -math.copysign(min(abs(v_perp) / dt, self.max_linear_accel), v_perp)
            return a_perp
        else:
            # Define d as the absolute distance to cover, and convert v_perp to the "correcting" frame.
            # In the correcting frame, positive means moving toward the trajectory.
            s = abs(perp_dist)
            # If perp_dist > 0, then the desired correction is to move in the negative v_perp direction.
            # Thus, we define:
            v_corr = -v_perp * sign_perp_dist
            # Now, v_corr > 0 means we are moving toward the trajectory, v_corr < 0 means moving away.

            # Calculate d but prevent division by zero
            if s < epsilon:
                d = 0  # Set to 0 or use a default minimal value
            else:
                d = v_perp**2 / (2 * s)
                
            if d > epsilon and abs(v_perp / d) < t_parallel and v_corr > 0:
                a_perp = d * sign_perp_dist
            elif v_corr > 0 and stopping_dist >= abs(perp_dist):
                a_perp = sign_perp_dist * self.max_linear_accel
                if self.debug_mode:
                    self.get_logger().debug(f"Full perpendicular correction: dist={abs(perp_dist):.3f}, stopping_dist={stopping_dist:.3f}")
            else:
                # Solve the quadratic:
                #    t_parallel^2 * a^2 - (4*d - 2*v_corr*t_parallel)*a - v_corr^2 = 0
                A = t_parallel ** 2
                B = -(4 * s - 2 * v_corr * t_parallel)
                C = -v_corr**2

                disc = B**2 - 4 * A * C
                if disc < 0:
                    # In case of numerical issues, fall back to max acceleration
                    a_perp = -self.max_linear_accel * sign_perp_dist
                else:
                    sqrt_disc = math.sqrt(disc)
                    a1 = (-B + sqrt_disc) / (2 * A)
                    a2 = (-B - sqrt_disc) / (2 * A)
                    # We choose the smallest positive solution
                    candidates = [a for a in (a1, a2) if a > 0]
                    candidate_a = min(candidates) if candidates else self.max_linear_accel
                    if candidate_a <= self.max_linear_accel:
                        a_perp = -candidate_a * sign_perp_dist  # because accelerating towards trajectory
                    else:
                        # Relax the available time t_parallel so that a_req does not exceed max_linear_accel
                        t_parallel = (-v_corr + math.sqrt(2 * (v_corr**2 + 2 * d * self.max_linear_accel))) / self.max_linear_accel
                        if self.debug_mode:
                            self.get_logger().debug(f"Relaxing time: new time={t_parallel:.3f}")
                        a_perp = -self.max_linear_accel * sign_perp_dist if abs(v_corr) < self.max_linear_velocity else 0.0
        
        # Clamp the result between -max_linear_accel and +max_linear_accel
        a_perp = self.clamp(a_perp, -self.max_linear_accel, self.max_linear_accel)

        if self.debug_mode:
            self.get_logger().debug(
                f'PERP: dist={perp_dist:.3f}, v={v_perp:.3f}, a={a_perp:.3f}'
            )

        return a_perp
        
    def calculate_velocity_commands(self, a_parallel, a_perp, dt):
        """Calculate new velocity commands based on accelerations"""
        # Convert accelerations back to global frame
        accel_x = a_parallel * self.unit_parallel[0] + a_perp * self.unit_perp[0]
        accel_y = a_parallel * self.unit_parallel[1] + a_perp * self.unit_perp[1]
        
        print(f"\naccerations = (parallel = {a_parallel}, perp = {a_perp}, total = {math.sqrt(a_parallel**2 + a_perp**2)})\n")

        # Apply accelerations to the last commanded velocities
        new_vx = self.last_cmd_vx + accel_x * dt
        new_vy = self.last_cmd_vy + accel_y * dt
        
        # Calculate total velocity magnitude for damping
        vel_magnitude = math.sqrt(new_vx**2 + new_vy**2)
        
        # Apply velocity magnitude limits
        if vel_magnitude > self.max_linear_velocity:
            scale_factor = self.max_linear_velocity / vel_magnitude
            new_vx *= scale_factor
            new_vy *= scale_factor
        
        return new_vx, new_vy
        
    def clamp(self, value, min_val, max_val):
        """Clamp a value between min and max"""
        return max(min(value, max_val), min_val)
        
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    controller = TrajectoryController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()