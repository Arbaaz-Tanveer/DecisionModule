#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import pygame
import time


class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')

        self.current_robot = 1  # Start with b1
        self.vel_publisher = self.create_publisher(Twist, f'b{self.current_robot}/cmd_vel', 10)
        self.kick_publisher = self.create_publisher(String, '/simulation/command', 10)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("Keyboard Controller")

        self.kick_press_time = {}
        self.timer = self.create_timer(0.02, self.process_input)

        self.get_logger().info("Keyboard control ready (WASD/arrow keys to move, K/L to kick, 1–5 to switch robot)")

    def calculate_kick_velocity(self, press_duration):
        return min(800, 200 + press_duration * 300)

    def switch_robot(self, robot_number):
        self.current_robot = robot_number
        self.vel_publisher = self.create_publisher(Twist, f'b{self.current_robot}/cmd_vel', 10)
        self.get_logger().info(f"Switched control to b{self.current_robot}")

    def process_input(self):
        pygame.event.pump()  # Process internal events
        keys = pygame.key.get_pressed()

        # Movement
        twist = Twist()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            twist.linear.y = 2.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            twist.linear.y = -2.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            twist.linear.x = -2.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            twist.linear.x = 2.0
        if keys[pygame.K_q]:
            twist.angular.z = 2.0
        if keys[pygame.K_e]:
            twist.angular.z = -2.0

        self.vel_publisher.publish(twist)

        # Handle events (kick + robot switch)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_k, pygame.K_l]:
                    self.kick_press_time[event.key] = time.time()

                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    robot_number = event.key - pygame.K_0
                    self.switch_robot(robot_number)

            if event.type == pygame.KEYUP and event.key in [pygame.K_k, pygame.K_l]:
                duration = time.time() - self.kick_press_time.get(event.key, time.time())
                velocity = self.calculate_kick_velocity(duration)
                angle = 30 if event.key == pygame.K_k else 0

                kick_cmd = String()
                kick_cmd.data = f"KICK {int(velocity)} {angle}"
                self.kick_publisher.publish(kick_cmd)


def main():
    rclpy.init()
    controller = KeyboardController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()
    pygame.quit()


if __name__ == '__main__':
    main()