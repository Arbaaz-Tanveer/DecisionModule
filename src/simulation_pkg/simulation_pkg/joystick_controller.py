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
        for i in range(min(len(self.joysticks), 5)):
            topic = f'b{i+1}/cmd_vel'
            self.vel_publishers[i] = self.create_publisher(Twist, topic, 10)
            
        self.get_logger().info(f'Initialized with {len(self.joysticks)} joysticks')
            
        # Button press timing for velocity calculation
        self.button_press_times = {i: 0.0 for i in range(len(self.joysticks))}
        
        # Start processing joystick input
        self.timer = self.create_timer(0.02, self.process_joystick_input)

    def calculate_kick_velocity(self, press_duration):
        base_velocity = 200
        velocity = base_velocity + (press_duration * 300)
        return min(800, velocity)

    def process_joystick_input(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in [10, 11]:
                    self.button_press_times[event.joy] = time.time()
                    
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [10, 11]:
                    press_duration = time.time() - self.button_press_times[event.joy]
                    velocity = self.calculate_kick_velocity(press_duration)
                    angle = 30 if event.button == 10 else 0
                    kick_cmd = String()
                    kick_cmd.data = f"KICK {int(velocity)} {angle}"
                    self.kick_publisher.publish(kick_cmd)

        # Process movement only for available joysticks
        for i in range(len(self.joysticks)):
            if i >= 5:  # Don't exceed 5 robots
                break
                
            joy = self.joysticks[i]
            twist = Twist()
            twist.linear.x = joy.get_axis(0) * 2.0
            twist.linear.y = -joy.get_axis(1) * 2.0
            twist.angular.z = -joy.get_axis(2) * 2.0
            
            self.vel_publishers[i].publish(twist)

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
