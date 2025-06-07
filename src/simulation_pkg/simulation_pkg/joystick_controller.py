#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import pygame
import time

class JoystickController(Node):
    def __init__(self):
        super().__init__('joystick_controller')
        
        # Velocity and acceleration limits
        self.MAX_LINEAR_VEL = 2.0      # m/s
        self.MAX_ANGULAR_VEL = 2.0     # rad/s
        self.MAX_LINEAR_ACCEL = 2.0    # m/s²
        self.MAX_ANGULAR_ACCEL = 2.0   # rad/s²
        
        # Initialize publishers for each robot
        self.vel_publishers = {}
        self.kick_publisher = self.create_publisher(String, '/simulation/command', 10)
        
        # Initialize pygame and joysticks
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        self.joysticks = []
        for i in range(joystick_count):
            joy = pygame.joystick.Joystick(i)
            joy.init()
            self.joysticks.append(joy)
            
        # Only create publishers for available joysticks
        for i in range(5):
            topic = f'b{i+1}/cmd_vel'
            self.vel_publishers[i] = self.create_publisher(Twist, topic, 10)
            
        self.get_logger().info(f'Initialized with {len(self.joysticks)} joysticks')
        
        # Button press timing for velocity calculation
        self.button_press_times = {i: 0.0 for i in range(len(self.joysticks))}
        
        # Current and target velocities for each robot
        self.current_velocities = {}
        self.target_velocities = {}
        for i in range(len(self.joysticks)):
            self.current_velocities[i] = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
            self.target_velocities[i] = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        
        # Timer period
        self.dt = 0.02  # 50 Hz
        
        # Start processing joystick input
        self.timer = self.create_timer(self.dt, self.process_joystick_input)
        self.selected_robot = {i: i for i in range(len(self.joysticks))}

    def limit_velocity(self, velocity, max_vel):
        """Limit velocity to maximum allowed value"""
        if abs(velocity) > max_vel:
            return max_vel if velocity > 0 else -max_vel
        return velocity

    def apply_acceleration_limit(self, current_vel, target_vel, max_accel, dt):
        """Apply acceleration limit to velocity change"""
        vel_diff = target_vel - current_vel
        max_vel_change = max_accel * dt
        
        if abs(vel_diff) > max_vel_change:
            if vel_diff > 0:
                return current_vel + max_vel_change
            else:
                return current_vel - max_vel_change
        else:
            return target_vel

    def calculate_kick_velocity(self, press_duration):
        base_velocity = 200
        velocity = base_velocity + (press_duration * 300)
        return min(800, velocity)

    def process_joystick_input(self):
        # Handle pygame events
        for event in pygame.event.get():
            # 1) Cycle which bot this joystick controls
            if event.type == pygame.JOYBUTTONDOWN and event.button == 1:
                old = self.selected_robot[event.joy]
                new = (old + 1) % len(self.vel_publishers)
                self.selected_robot[event.joy] = new
                self.get_logger().info(f"Joystick {event.joy}: switched from bot {old} to bot {new}")
                continue

            # 2) Kick button logic (buttons 10 & 11)
            if event.type == pygame.JOYBUTTONDOWN and event.button in [10, 11]:
                self.button_press_times[event.joy] = time.time()
            elif event.type == pygame.JOYBUTTONUP and event.button in [10, 11]:
                press_duration = time.time() - self.button_press_times[event.joy]
                velocity = self.calculate_kick_velocity(press_duration)
                angle = 30 if event.button == 10 else 0
                kick_cmd = String()
                kick_cmd.data = f"KICK {int(velocity)} {angle}"
                self.kick_publisher.publish(kick_cmd)

        # Process movement for each joystick
        for joy_idx, joy in enumerate(self.joysticks):
            if joy_idx not in self.selected_robot:
                continue  # safety guard

            # Raw axes → target velocities
            tx = self.limit_velocity(joy.get_axis(0) * self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
            ty = self.limit_velocity(-joy.get_axis(1) * self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
            tz = self.limit_velocity(-joy.get_axis(2) * self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

            # Store targets
            self.target_velocities[joy_idx]['linear_x']  = tx
            self.target_velocities[joy_idx]['linear_y']  = ty
            self.target_velocities[joy_idx]['angular_z'] = tz

            # Apply acceleration limits
            cvx = self.apply_acceleration_limit(
                self.current_velocities[joy_idx]['linear_x'], tx,
                self.MAX_LINEAR_ACCEL, self.dt
            )
            cvy = self.apply_acceleration_limit(
                self.current_velocities[joy_idx]['linear_y'], ty,
                self.MAX_LINEAR_ACCEL, self.dt
            )
            cvz = self.apply_acceleration_limit(
                self.current_velocities[joy_idx]['angular_z'], tz,
                self.MAX_ANGULAR_ACCEL, self.dt
            )

            # Update current velocities
            self.current_velocities[joy_idx]['linear_x']  = cvx
            self.current_velocities[joy_idx]['linear_y']  = cvy
            self.current_velocities[joy_idx]['angular_z'] = cvz

            # Build and publish Twist to the selected bot
            twist = Twist()
            twist.linear.x  = cvx
            twist.linear.y  = cvy
            twist.angular.z = cvz

            bot_id = self.selected_robot[joy_idx]
            self.vel_publishers[bot_id].publish(twist)

def main():
    rclpy.init()
    controller = JoystickController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
        
    controller.destroy_node()
    rclpy.shutdown()
    pygame.quit()

if __name__ == '__main__':
    main()