import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry 
from std_msgs.msg import String, Float32, Float32MultiArray, MultiArrayDimension
import pygame
import math
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

# Adding soccer ball constants
BALL_DIAMETER = 22  # cm
BALL_MASS = 0.4  # kg
GRAVITY = 980  # cm/s²
AIR_RESISTANCE = 0.1
GROUND_DAMPING = 0.7
CATCH_RADIUS = 30  # cm from center of robot
POSSESSION_COOLDOWN = 0.5  # seconds after kick before allowing new possession
MAX_KICK_SPEED = 800  # cm/s
MIN_KICK_ANGLE = 0  # degrees from horizontal
MAX_KICK_ANGLE = 45  # degrees from horizontal

# Constants for the simulation
FIELD_WIDTH = 2200  # 22 meters in centimeters
FIELD_HEIGHT = 1400  # 14 meters in centimeters
WHEEL_DIAMETER = 10  # cm
BOT_SIZE = 50  # cm
TICKS_PER_REVOLUTION = 1750
BOT_MASS = 20  # kg
WHEEL_DISTANCE = BOT_SIZE / math.sqrt(2)  # Distance from center to wheel
MOMENT_OF_INERTIA = BOT_MASS * ((BOT_SIZE/100) ** 2) / 6  # Approximating as solid square
FPS = 40
MAX_MOTOR_FORCE = 40  # Newtons
KP = 2.0
KI = 0
KD = 0.5
MAX_TICKS_PER_SECOND = 30000
TICKS_PER_METER = (100 * TICKS_PER_REVOLUTION) / (math.pi*WHEEL_DIAMETER)

@dataclass
class BallState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    possessed_by: int = -1
    possession_cooldown: float = 0

class SoccerBall:
    def __init__(self, x: float, y: float):
        self.state = BallState(x=x, y=y, z=0, vx=0, vy=0, vz=0)
        
    def update(self, dt: float, robots: List['OmniwheelRobot']) -> None:
        if self.state.possession_cooldown > 0:
            self.state.possession_cooldown -= dt
            
        if self.state.possessed_by >= 0:
            # Update position with robot
            robot = robots[self.state.possessed_by]
            self.state.x = robot.x + CATCH_RADIUS * math.cos(robot.theta)
            self.state.y = robot.y + CATCH_RADIUS * math.sin(robot.theta)
            self.state.z = 0
            self.state.vx = robot.vx
            self.state.vy = robot.vy
            self.state.vz = 0
            return
            
        # Apply physics when not possessed
        # Update velocities with gravity and air resistance
        self.state.vx *= (1 - AIR_RESISTANCE * dt)
        self.state.vy *= (1 - AIR_RESISTANCE * dt)
        self.state.vz -= GRAVITY * dt
        self.state.vz *= (1 - AIR_RESISTANCE * dt)
        
        # Update positions
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.z += self.state.vz * dt
        
        # Ground collision
        if self.state.z < 0:
            self.state.z = 0
            self.state.vz = -self.state.vz * GROUND_DAMPING

        if self.state.z<0.1:
            self.state.vx *= (1-(1-GROUND_DAMPING) * dt)
            self.state.vy *= (1-(1-GROUND_DAMPING) * dt)

        # Check for robot possession
        if self.state.z < BALL_DIAMETER/2:  # Only allow possession when ball is near ground
            for i, robot in enumerate(robots):
                dx = self.state.x - robot.x
                dy = self.state.y - robot.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < CATCH_RADIUS and self.state.possession_cooldown <= 0:
                    # Check if ball is in front of robot (white circle area)
                    relative_angle = math.atan2(dy, dx) - robot.theta
                    if abs(relative_angle) < math.pi/4:  # 45 degree catch zone
                        self.state.possessed_by = i
                        break
                elif dist < BOT_SIZE:  # Bounce off robot
                    # Calculate bounce direction
                    bounce_angle = math.atan2(dy, dx)
                    speed = math.sqrt(self.state.vx**2 + self.state.vy**2)
                    self.state.vx = speed * math.cos(bounce_angle) * GROUND_DAMPING
                    self.state.vy = speed * math.sin(bounce_angle) * GROUND_DAMPING
        
        # Field boundaries
        if self.state.x < 0:
            self.state.x = 0
            self.state.vx = -self.state.vx * GROUND_DAMPING
        elif self.state.x > FIELD_WIDTH:
            self.state.x = FIELD_WIDTH
            self.state.vx = -self.state.vx * GROUND_DAMPING
            
        if self.state.y < 0:
            self.state.y = 0
            self.state.vy = -self.state.vy * GROUND_DAMPING
        elif self.state.y > FIELD_HEIGHT:
            self.state.y = FIELD_HEIGHT
            self.state.vy = -self.state.vy * GROUND_DAMPING

    def kick(self, speed: float, vertical_angle: float, robots: List['OmniwheelRobot']) -> None:
        """
        Kick the ball with given speed and vertical angle
        speed: cm/s
        vertical_angle: degrees from ground
        """
        if self.state.possessed_by < 0:
            return
            
        # Get the robot's direction
        robot = robots[self.state.possessed_by]
        robot_angle = robot.theta
        
        # Convert vertical angle to radians and clamp it
        vertical_rad = math.radians(np.clip(vertical_angle, MIN_KICK_ANGLE, MAX_KICK_ANGLE))
        
        # Calculate velocity components
        speed = min(speed, MAX_KICK_SPEED)
        horizontal_speed = speed * math.cos(vertical_rad)
        
        # Set velocities based on robot's direction
        self.state.vx = horizontal_speed * math.cos(robot_angle)
        self.state.vy = horizontal_speed * math.sin(robot_angle)
        self.state.vz = speed * math.sin(vertical_rad)
        
        self.state.possessed_by = -1
        self.state.possession_cooldown = POSSESSION_COOLDOWN
        

    def draw(self, screen: pygame.Surface, scale: float) -> None:
        # Draw ball shadow
        ball_x = int(self.state.x * scale)
        ball_y = int(self.state.y * scale)
        ball_radius = int((BALL_DIAMETER/2) * scale)
        
        # Draw actual ball with slight offset based on height
        shadow_x = ball_x
        shadow_y = int(ball_y - self.state.z * scale)
        pygame.draw.circle(screen, (105, 105, 105), (shadow_x, shadow_y), ball_radius)

        pygame.draw.circle(screen, (255, 255, 0), (ball_x, ball_y), ball_radius)

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, target: float, current: float, dt: float) -> float:
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class OmniwheelRobot:
    def __init__(self, robot_number: int, x: float, y: float, theta: float = 0, color: tuple = (255, 0, 0)):
        self.robot_number = robot_number
        self.x = x
        self.y = y
        self.theta = theta  # radians
        self.color = color  # Added color parameter
        
        # Linear and angular velocities
        self.vx = 0  # cm/s
        self.vy = 0  # cm/s
        self.omega = 0  # rad/s
        
        self.target_vx = 0
        self.target_vy = 0
        self.target_omega = 0
        
        # Linear and angular accelerations
        self.ax = 0  # cm/s²
        self.ay = 0  # cm/s²
        self.alpha = 0  # rad/s²
        
        # Wheel states
        self.desired_ticks = [0, 0, 0, 0]  # Cumulative desired ticks
        self.achieved_ticks = [0, 0, 0, 0]  # Cumulative achieved ticks
        self.current_wheel_velocities = [0, 0, 0, 0]  # Current velocities in ticks/second
        
        # PID controllers for each wheel
        self.pid_controllers = [
            PIDController(KP, KI, KD) for _ in range(4)
        ]
        
        # Wheel angles (45°, 135°, 225°, 315° from x-axis)
        self.wheel_angles = [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]   #FL,FR,BR,BL
        
        # Motor forces
        self.motor_forces = [0, 0, 0, 0]
        
        # Target velocities for visualization
        self.target_velocities = [0, 0, 0, 0]

    def set_velocity(self, vx: float, vy: float, omega: float):
        # Calculate wheel velocities from stored values
        wheel_velocities = [
            vx + vy + omega,  # FR
            -vx + vy + omega,  # FL
            -vx - vy + omega,  # BL
            vx - vy + omega   # BR
        ]
        self.add_target_ticks(wheel_velocities)
    
    def add_target_ticks(self, ticks_per_second: List[float]):
        """Add desired ticks per second to the cumulative desired ticks"""
        self.target_velocities = ticks_per_second  # Store for visualization
        for i in range(4):
            self.desired_ticks[i] += ticks_per_second[i] / FPS  # Convert to ticks per frame
    
    def update(self, dt: float):
        """Update robot state based on physics"""
        # Calculate motor forces using PID controllers
        self.motor_forces = []
        for i in range(4):
            pid_output = self.pid_controllers[i].compute(
                self.desired_ticks[i],
                self.achieved_ticks[i],
                dt
            )
            force = np.clip(pid_output*MAX_MOTOR_FORCE/255, -MAX_MOTOR_FORCE, MAX_MOTOR_FORCE)
            self.motor_forces.append(force)
        
        # Calculate net forces and torque in robot frame
        fx = fy = 0
        torque = 0
        for i, force in enumerate(self.motor_forces):
            angle = self.wheel_angles[i]
            fx += force * math.cos(angle)
            fy += force * math.sin(angle)
            torque += force * WHEEL_DISTANCE/100
        
        # Calculate accelerations
        self.ax = 100*fx / BOT_MASS
        self.ay = 100*fy / BOT_MASS
        self.alpha = torque / MOMENT_OF_INERTIA
        
        # Update velocities
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.omega += self.alpha * dt

        ''' self.vx = fx
        self.vy = fy
        self.omega = torque / MOMENT_OF_INERTIA '''
        
        # Update position and orientation
        world_vx = self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)
        world_vy = self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)
        
        self.x += world_vx * dt
        self.y += world_vy * dt
        self.theta += self.omega * dt
        
        # Calculate current wheel velocities and update achieved ticks
        wheel_radius = WHEEL_DIAMETER / 2
        for i in range(4):
            angle = self.wheel_angles[i]
            linear_velocity = (self.vx * math.cos(angle) + self.vy * math.sin(angle) 
                             + self.omega * WHEEL_DISTANCE)
            self.current_wheel_velocities[i] = (linear_velocity * TICKS_PER_REVOLUTION 
                                              / (2 * math.pi * wheel_radius))
            self.achieved_ticks[i] += self.current_wheel_velocities[i] * dt
        
        # Keep theta between -pi and pi
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def draw(self, screen: pygame.Surface, scale: float):
        """Draw the robot on the screen"""
        px = int(self.x * scale)
        py = int(self.y * scale)
        size = int(BOT_SIZE * scale)
        
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surface, (*self.color, 128), (0, 0, size, size))  # Using robot's color
        pygame.draw.circle(surface, (255,255,255), (size, size/2), size/4, 0)
        
        # Add robot number
        font = pygame.font.Font(None, int(size/1.5))
        number_text = font.render(str(self.robot_number), True, (255, 255, 255))
        text_rect = number_text.get_rect(center=(size/2, size/2))
        surface.blit(number_text, text_rect)

        wheel_size = int(WHEEL_DIAMETER * scale)
        for i, angle in enumerate(self.wheel_angles):
            wheel_x = size/2 + WHEEL_DISTANCE*scale*math.cos(angle)
            wheel_y = size/2 - WHEEL_DISTANCE*scale*math.sin(angle)
            
            velocity_ratio = self.target_velocities[i] / MAX_TICKS_PER_SECOND
            if velocity_ratio > 0:
                color = (255, 0, 0)
            elif velocity_ratio < 0:
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
            
            pygame.draw.circle(surface, (0,0,0), (int(wheel_x), int(wheel_y)), wheel_size)
        
        rotated = pygame.transform.rotate(surface, -math.degrees(self.theta))
        screen.blit(rotated, (px - rotated.get_width()/2, py - rotated.get_height()/2))

class ROS2RobotSimulation(Node):
    def __init__(self):
        super().__init__('robot_simulation')
        
        # Initialize Pygame
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ROS2 Omniwheel Robot Simulation")
        self.scale = min(self.width/FIELD_WIDTH, self.height/FIELD_HEIGHT)
        self.clock = pygame.time.Clock()
        
        # Initialize robot instances and their publishers/subscribers
        self.num_robots = 10  # Total number of robots (5 per team)
        self.robots = []
        self.cmd_vel_subs = []
        self.state_pubs = []
        self.odom_pubs = []
        
        # Create team-based namespaces
        self.robot_namespaces = []
        for team in ['o', 'b']:
            for i in range(1, 6):  # 5 robots per team
                self.robot_namespaces.append(f'{team}{i}')
        
        # Define colors for teams (Team A: Red-based, Team B: Blue-based)
        team_colors = [
            # Team A colors (red-based)
            (255, 0, 0),      # Pure red
            (255, 0, 0),    
            (255, 0, 0),   
            (255, 0, 0),  
            (255, 0, 0),  
            # Team B colors (blue-based)
            (0, 0, 255),      # Pure blue
            (0, 0, 255),  
            (0, 0, 255),   
            (0, 0, 255),
            (0, 0, 255)   
        ]
        
        # Create robots and their publishers/subscribers
        for i in range(self.num_robots):
            # Determine team and position
            is_team_b = i >= 5
            team_offset = 1 if is_team_b else -1
            
            # Calculate starting positions
            row = (i % 5) // 3  # 0 or 1 for two rows
            col = (i % 5) % 3   # 0 to 2 for three columns
            
            # Position team A on left side, team B on right side
            x = FIELD_WIDTH/4 * (2 + team_offset) + col * 150  # Adjust multiplier for spread
            y = FIELD_HEIGHT/2 -100 + row * 200
            
            # Create robot instance
            robot = OmniwheelRobot((i%5) + 1, x, y, color=team_colors[i])
            self.robots.append(robot)
            
            namespace = self.robot_namespaces[i]
            
            # Command velocity subscriber
            self.cmd_vel_subs.append(
                self.create_subscription(
                    Twist,
                    f'{namespace}/cmd_vel',
                    lambda msg, idx=i: self.cmd_vel_callback(msg, idx),
                    10
                )
            )
            
            # Odometry publisher
            self.odom_pubs.append(
                self.create_publisher(
                    Odometry,
                    f'{namespace}/odom',
                    10
                )
            )
            
            # State publisher
            self.state_pubs.append(
                self.create_publisher(
                    Float32MultiArray,
                    f'{namespace}_data',
                    10
                )
            )
        # Add ball instance
        self.ball = SoccerBall(FIELD_WIDTH/4, FIELD_HEIGHT/4)
        
        #ball state publisher
        self.ball_state_pub = self.create_publisher(
            Float32MultiArray,
            'ball_data',
            10
        )
        
        # Simplified command subscriber
        self.command_sub = self.create_subscription(
            String,
            'simulation/command',
            self.command_callback,
            10
        )
        # Create global string publishers and subscribers
        self.status_pub = self.create_publisher(String, 'simulation/status', 10)

        # Create timer for updating and rendering
        self.create_timer(1.0/FPS, self.update_and_render)
    
    # Modified command callback to handle ball commands
    def command_callback(self, msg: String):
        """Handle simplified command format for ball control"""
        try:
            cmd_parts = msg.data.split()
            
            if cmd_parts[0] == "KICK":
                if len(cmd_parts) == 3 and self.ball.state.possessed_by >= 0:
                    speed = float(cmd_parts[1])
                    angle = float(cmd_parts[2])
                    self.ball.kick(speed, angle, self.robots)
            
            elif cmd_parts[0] == "PLACE":
                if len(cmd_parts) == 4:
                    x = float(cmd_parts[1]) * 100  # Convert to cm
                    y = float(cmd_parts[2]) * 100
                    z = float(cmd_parts[3]) * 100
                    self.ball.state.x = FIELD_WIDTH/2 + x
                    self.ball.state.y = FIELD_HEIGHT/2 - y
                    self.ball.state.z = z
                    self.ball.state.vx = 0
                    self.ball.state.vy = 0
                    self.ball.state.vz = 0
                    self.ball.state.possessed_by = -1
                    
        except (IndexError, ValueError) as e:
            self.get_logger().warn(f'Invalid command format: {e}')
    
    def publish_ball_state(self):
        """Publish ball state as Float32MultiArray with proper type validation"""
        state_msg = Float32MultiArray()
        
        # Define dimensions
        state_msg.layout.dim = [MultiArrayDimension()]
        state_msg.layout.dim[0].label = "state"
        state_msg.layout.dim[0].size = 7  # x, y, z, vx, vy, vz, possession
        state_msg.layout.dim[0].stride = 7
        
        # Convert all values to float32 and ensure they're within valid ranges
        try:
            state_data = [
                float(max(min((self.ball.state.x - FIELD_WIDTH/2)/100, 3.4e38), -3.4e38)),  # x position
                float(max(min((FIELD_HEIGHT/2 - self.ball.state.y)/100, 3.4e38), -3.4e38)),  # y position
                float(max(min(self.ball.state.z/100, 3.4e38), -3.4e38)),  # z position
                float(max(min(self.ball.state.vx/100, 3.4e38), -3.4e38)),  # x velocity
                float(max(min(self.ball.state.vy/100, 3.4e38), -3.4e38)),  # y velocity
                float(max(min(self.ball.state.vz/100, 3.4e38), -3.4e38)),  # z velocity
                float(max(min(self.ball.state.possessed_by, 3.4e38), -3.4e38))  # possession
            ]
            state_msg.data = state_data
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'Error converting ball state data: {e}')
            # Provide safe default values if conversion fails
            state_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        
        self.ball_state_pub.publish(state_msg)


    def update_and_render(self):
        """Combined update and render callback"""
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                return
        
        # Update robot states
        dt = 1.0/FPS
        for i, robot in enumerate(self.robots):
            robot.set_velocity(robot.target_vx, robot.target_vy, robot.target_omega)
            robot.update(dt)
            self.publish_robot_state(robot, i)
        
        # Publish status
        status_msg = String()
        status_msg.data = "Simulation running"
        self.status_pub.publish(status_msg)
        

        # Update ball physics
        self.ball.update(1.0/FPS, self.robots)
        
        # Publish ball state using new format
        self.publish_ball_state()
        
        # Draw everything
        self.draw_field()
        for robot in self.robots:
            robot.draw(self.screen, self.scale)
        self.ball.draw(self.screen, self.scale)
            
        # Draw height bar
        bar_height = 200
        bar_width = 20
        bar_x = self.width - 40
        bar_y = self.height - 20 - bar_height
        max_height = 300  # cm
        
        # Background bar
        pygame.draw.rect(self.screen, (64, 64, 64),
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Height indicator
        height_percentage = min(self.ball.state.z / max_height, 1.0)
        fill_height = int(bar_height * height_percentage)
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (bar_x, bar_y + bar_height - fill_height,
                         bar_width, fill_height))
        
        # Height text
        font = pygame.font.Font(None, 24)
        height_text = font.render(f'Ball:{self.ball.state.z:.1f} cm',
                                True, (255, 255, 255))
        self.screen.blit(height_text,
                        (bar_x - 60, bar_y + bar_height - fill_height))
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def draw_field(self):
            """Draw the field with boundaries"""
            self.screen.fill((0, 100, 0))
            
            field_width_px = int(FIELD_WIDTH * self.scale)
            field_height_px = int(FIELD_HEIGHT * self.scale)
            start_x = (self.width - field_width_px) // 2
            start_y = (self.height - field_height_px) // 2
            
            pygame.draw.rect(self.screen, (255, 255, 255),
                            (start_x, start_y, field_width_px, field_height_px), 2)
            
            pygame.draw.line(self.screen, (255, 255, 255),
                            (self.width//2, start_y),
                            (self.width//2, start_y + field_height_px), 2)
            pygame.draw.circle(self.screen, (255, 255, 255),
                            (self.width//2, self.height//2),
                            int(50 * self.scale), 2)
    
    
    def cmd_vel_callback(self, msg: Twist, robot_index: int):
        """
        Convert Twist message from global frame velocities to robot frame velocities
        and convert to encoder ticks
        
        msg.linear.x: Global X velocity in m/s
        msg.linear.y: Global Y velocity in m/s
        msg.angular.z: Angular velocity in rad/s
        """
        # Get current robot orientation
        theta = self.robots[robot_index].theta
        
        # Convert global velocities to local frame using rotation matrix
        # [vx_local] = [ cos(θ)  sin(θ)] [vx_global]
        # [vy_local] = [-sin(θ)  cos(θ)] [vy_global]
        global_vx = msg.linear.x * TICKS_PER_METER  # Convert m/s to ticks/s
        global_vy = -msg.linear.y * TICKS_PER_METER
        
        local_vx = global_vx * math.cos(theta) + global_vy * math.sin(theta)
        local_vy = -global_vx * math.sin(theta) + global_vy * math.cos(theta)
        omega = -msg.angular.z * WHEEL_DISTANCE * TICKS_PER_METER/100
        
        # Store transformed velocities in robot
        self.robots[robot_index].target_vx = local_vx
        self.robots[robot_index].target_vy = local_vy  
        self.robots[robot_index].target_omega = omega  

        
    
    def publish_robot_state(self, robot: OmniwheelRobot, robot_index: int):
        """Publish robot state as Odometry and Position messages"""
        current_time = self.get_clock().now().to_msg()
        
        # Publish Odometry
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.child_frame_id = f"robot_{robot_index+1}_base_link"
        odom.header.stamp = current_time
        
        # Convert centimeters to meters for odometry
        odom.pose.pose.position.x = robot.x / 100.0
        odom.pose.pose.position.y = robot.y / 100.0
        odom.pose.pose.orientation.z = math.sin(robot.theta/2)
        odom.pose.pose.orientation.w = math.cos(robot.theta/2)
        
        odom.twist.twist.linear.x = robot.vx / 100.0
        odom.twist.twist.linear.y = -robot.vy / 100.0
        odom.twist.twist.angular.z = -robot.omega
        
        self.odom_pubs[robot_index].publish(odom)
        
        # Create Float32MultiArray message
        state_msg = Float32MultiArray()
        
        # Define dimensions
        state_msg.layout.dim = [MultiArrayDimension()]
        state_msg.layout.dim[0].label = "state"
        state_msg.layout.dim[0].size = 7  # x, y, theta, 4 encoder values
        state_msg.layout.dim[0].stride = 7
        
        # Fill data array [x, y, theta, encoder1, encoder2, encoder3, encoder4]
        state_msg.data = [
            (robot.x - FIELD_WIDTH/2)/100,                    # x position in cm
            (FIELD_HEIGHT/2 - robot.y)/100,                    # y position in cm
            -robot.theta,                # theta in radians
            -robot.achieved_ticks[0],    # Front left encoder
            -robot.achieved_ticks[3],    # Back left encoder
            -robot.achieved_ticks[1],    # Front right encoder
            -robot.achieved_ticks[2]     # Back right encoder
        ]
        
        # Publish state message
        self.state_pubs[robot_index].publish(state_msg)

        
def main(args=None):
    rclpy.init(args=args)
    sim_node = ROS2RobotSimulation()
    
    try:
        rclpy.spin(sim_node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        rclpy.shutdown()

if __name__ == '__main__':
    main()