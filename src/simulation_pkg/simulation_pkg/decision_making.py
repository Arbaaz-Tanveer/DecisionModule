import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String
from nav_msgs.msg import Odometry
import math
from math import atan2, sqrt
from .heat_maps import HeatMapClusterer, HeatMapGenerator, RoboCupState, HeatMapVisualizer
import cv2
from .robot_sim import BOT_SIZE
import time

PASS_ALIGNMENT_THRESHOLD = 0.1   #degreee of alignment in radians

GLOBAL_PROBABILITIES = {
    # Pass Probability Components
    'pass_distance_factor': 0.0,   # Distance between bots
    'pass_line_obstruction': 0.0,  # Opponent interference 
    'pass_heat_map_factor': 0.0,   # Strategic map value
    'final_pass_probability': 0.0,

    # Goal Probability Components
    'goal_distance_factor': 0.0,   # Distance to goal 
    'goal_heat_map_factor': 0.0,   # Strategic map value
    'goal_line_obstruction': 0.0,  # Opponent blocking
    'final_goal_probability': 0.0,

    # Additional future parameters can be added here
}


# --- Enums and Data Classes ---
class GameState(Enum):
    WE_HAVE_BALL = "we_have_ball"
    OPPONENT_HAS_BALL = "opponent_has_ball"
    LOOSE_BALL = "loose_ball"
    
class RobotRole(Enum):
    BALL_HANDLER = "ball_handler"
    ATTACKER = "attacker"
    MIDFIELDER = "midfielder"
    DEFENDER = "defender"
    GOALKEEPER = "goalkeeper"

@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0

@dataclass
class Velocity:
    vx: float
    vy: float
    vz: float = 0.0

# --- Robot, Ball and Field Classes ---
class Robot:
    def __init__(self, id: int, position: Position, velocity: Velocity, role: RobotRole):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.role = role
        self.target_position = Position(0.0, 0.0, 0.0)
        self.has_ball = False
        self.current_theta = 0.0
        self.target_theta = 0.0
        # Record the position where the robot first took possession of the ball.
        self.possession_position = None

class Ball:
    def __init__(self):
        self.position = Position(0.0, 0.0, 0.0)
        self.velocity = Velocity(0.0, 0.0, 0.0)
        self.possession = None  # None, 'our', or 'opponent'

class Field:
    def __init__(self):
        self.length = 22.0
        self.width = 14.0
        self.our_goal = Position(-self.length/2, 0.0)
        self.opponent_goal = Position(self.length/2, 0.0)

# --- Main Game Manager Node ---
class GameManagerROS2(Node):
    def __init__(self, state: RoboCupState, heat_generator: HeatMapGenerator, 
                 clusterer: HeatMapClusterer, visualizer: HeatMapVisualizer):
        super().__init__('decicion')
        self.field = Field()
        self.our_robots: List[Robot] = []
        self.opponent_robots: List[Robot] = []
        self.ball = Ball()
        self.game_state = GameState.LOOSE_BALL
        self.state = state
        self.heat_generator = heat_generator
        self.clusterer = clusterer
        self.visualizer = visualizer
        
        # --- Global pass-related state variables ---
        self.pass_in_progress = False      # True when a pass is underway.
        self.pass_start_time = 0           # Time at which the pass command was issued.
        self.pass_timeout = 0              # Allowed time for the pass transit.
        self.pass_receiver_id = None       # Intended receiver's robot ID.
        
        # --- New Publishers for command and pass status ---
        self.command_pub = self.create_publisher(String, 'simulation/command', 10)
        self.pass_status_pub = self.create_publisher(String, 'simulation/pass_status', 10)
        
        # Initialize robots with default values.
        self._initialize_robots()
        
        # Ball subscription.
        self.ball_sub = self.create_subscription(
            Float32MultiArray,
            'ball_data',
            self.ball_callback,
            10
        )
        
        # Robot position subscriptions.
        self.our_robot_subs = []
        self.opp_robot_subs = []
        for i in range(5):
            # Our robots.
            self.our_robot_subs.append(
                self.create_subscription(
                    Float32MultiArray,
                    f'o{i+1}_data',
                    lambda msg, idx=i: self.our_robot_callback(msg, idx),
                    10
                )
            )
            # Opponent robots.
            self.opp_robot_subs.append(
                self.create_subscription(
                    Float32MultiArray,
                    f'b{i+1}_data',
                    lambda msg, idx=i: self.opp_robot_callback(msg, idx),
                    10
                )
            )
        
        # Robot velocity subscriptions (odometry).
        self.our_robot_vel_subs = []
        self.opp_robot_vel_subs = []
        for i in range(5):
            # Our robots velocity.
            self.our_robot_vel_subs.append(
                self.create_subscription(
                    Odometry,
                    f'o{i+1}_odom',
                    lambda msg, idx=i: self.our_robot_vel_callback(msg, idx),
                    10
                )
            )
            # Opponent robots velocity.
            self.opp_robot_vel_subs.append(
                self.create_subscription(
                    Odometry,
                    f'b{i+1}_odom',
                    lambda msg, idx=i: self.opp_robot_vel_callback(msg, idx),
                    10
                )
            )
        
        # Publishers for target positions.
        self.target_pubs = []
        for i in range(5):
            self.target_pubs.append(
                self.create_publisher(
                    Float32MultiArray,
                    f'/o{i+1}/decision_target_data',
                    10
                )
            )
        
        # Timer for decision-making loop.
        self.timer = self.create_timer(0.1, self.decision_making_callback)
        self.get_logger().info('Game Manager Node initialized')

    #aligning the bots towards the ball
    def reset_bot_orientation(self):
        """Make all robots except ball handler face the ball."""
        ball_holder_id = self.state.ball_holder
        for robot in self.our_robots:
            if robot.id != ball_holder_id:
                # Calculate angle to ball
                dx = self.ball.position.x - robot.position.x
                dy = self.ball.position.y - robot.position.y
                robot.target_theta = math.atan2(dy, dx)

    # --- Callbacks for updating state ---
    def ball_callback(self, msg):
        """Handle ball data updates."""
        self.ball.position = Position(
            msg.data[0], msg.data[1], msg.data[2]
        )
        self.ball.velocity = Velocity(
            msg.data[3], msg.data[4], msg.data[5]
        )
        # state.ball_holder is assumed to be an integer (-1 if no one has possession).
        self.state.ball_holder = int(msg.data[6])
        self.state.ball_position = np.array([self.ball.position.x, self.ball.position.y])

    def our_robot_callback(self, msg, robot_idx):
        """Handle our robot position updates."""
        self.our_robots[robot_idx].position = Position(
            msg.data[0], msg.data[1], 0
        )
        self.our_robots[robot_idx].current_theta = msg.data[2]
        self.state.our_positions[robot_idx] = [msg.data[0], msg.data[1]]

    def opp_robot_callback(self, msg, robot_idx):
        """Handle opponent robot position updates."""
        self.opponent_robots[robot_idx].position = Position(
            msg.data[0], msg.data[1], msg.data[2]
        )
        self.state.opp_positions[robot_idx] = [msg.data[0], msg.data[1]]

    def our_robot_vel_callback(self, msg, robot_idx):
        """Handle our robot velocity updates."""
        self.our_robots[robot_idx].velocity = Velocity(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        )
        self.state.our_velocities[robot_idx] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        ]

    def opp_robot_vel_callback(self, msg, robot_idx):
        """Handle opponent robot velocity updates."""
        self.opponent_robots[robot_idx].velocity = Velocity(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        )
        self.state.opp_velocities[robot_idx] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y
        ]

    def publish_target_positions(self):
        """Publish target positions for all robots."""
        for i, robot in enumerate(self.our_robots):
            msg = Float32MultiArray()
            msg.data = [
                robot.target_position.x,
                robot.target_position.y,
                robot.target_theta
            ]
            self.target_pubs[i].publish(msg)

    def decision_making_callback(self):
        """Main decision-making loop triggered by timer."""
        # If a pass is in progress, do not override that with loose-ball logic.
        self.reset_bot_orientation()  #to align the bots towards the ball
        if not self.pass_in_progress:
            self.update_game_state()
            if self.game_state == GameState.WE_HAVE_BALL:
                self._handle_we_have_ball()
            elif self.game_state == GameState.OPPONENT_HAS_BALL:
                self._handle_opponent_has_ball()
            elif self.game_state == GameState.LOOSE_BALL:
                self._handle_loose_ball()
        else:
            self.get_logger().info("Pass in progress: skipping decision-making for ball control.")
        self.check_pass_status()
        self.publish_target_positions()

    def _initialize_robots(self):
        """Initialize robots with default positions."""
        roles = [RobotRole.GOALKEEPER, RobotRole.DEFENDER, RobotRole.MIDFIELDER, 
                 RobotRole.ATTACKER, RobotRole.ATTACKER]
        for i in range(5):
            pos = self.state.our_positions[i]
            vel = self.state.our_velocities[i]
            self.our_robots.append(
                Robot(i, Position(pos[0], pos[1], 0.0), 
                      Velocity(vel[0], vel[1], 0.0), roles[i])
            )
        for i in range(5):
            pos = self.state.opp_positions[i]
            vel = self.state.opp_velocities[i]
            self.opponent_robots.append(
                Robot(i, Position(pos[0], pos[1], 0.0),
                      Velocity(vel[0], vel[1], 0.0), RobotRole.ATTACKER)
            )

    def determine_ball_possession(self):
        """Determine which team has possession of the ball."""
        POSSESSION_THRESHOLD = 0.5  # meters
        opp_min_dist = float('inf')
        for robot in self.opponent_robots:
            dist = math.sqrt((robot.position.x - self.ball.position.x)**2 +
                             (robot.position.y - self.ball.position.y)**2)
            if dist < opp_min_dist:
                opp_min_dist = dist
        
        if self.state.ball_holder >= 0 and self.state.ball_holder < 5:
            return GameState.WE_HAVE_BALL
        elif opp_min_dist < POSSESSION_THRESHOLD:
            return GameState.OPPONENT_HAS_BALL
        else:
            return GameState.LOOSE_BALL

    def update_game_state(self):
        """Update the game state based on ball possession."""
        self.game_state = self.determine_ball_possession()
        self.get_logger().info(f'Current game state: {self.game_state}')

    def draw_assignment_lines(self, image):
        scale_factor = self.visualizer.scale  # match the scaling used in visualization
        for robot in self.our_robots:
            sy, sx = self.visualizer.get_nearest_index((robot.position.x, robot.position.y))
            ey, ex = self.visualizer.get_nearest_index((robot.target_position.x, robot.target_position.y))
            sx = int(sx * scale_factor)
            sy = int(sy * scale_factor)
            ex = int(ex * scale_factor)
            ey = int(ey * scale_factor)
            self.visualizer.draw_dotted_line(image, (sx, sy), (ex, ey), (255, 255, 255), thickness=3, gap=10)

    # --- Helper functions for goal and pass probabilities ---
    def calculate_goal_probability(self, bot: Robot) -> float:
        """
        Enhanced goal probability calculation:
        1. Distance to goal
        2. Heat map value
        3. Opponent interference
        """
        goal = self.field.opponent_goal
        
        # Distance factor
        dx = bot.position.x - goal.x
        dy = bot.position.y - goal.y
        distance = math.sqrt(dx*dx + dy*dy)
        max_distance = self.field.length
        distance_factor = 1.0 - (distance / max_distance)
        
        # Heat map factor
        goal_heat_map = self.heat_generator.goal_direction_map()
        heat_map_factor = goal_heat_map[int(bot.position.y), int(bot.position.x)] / 255.0
        
        # Opponent interference
        line_obstruction = self.calculate_pass_line_obstruction((bot.position.x,bot.position.y), (goal.x,goal.y))
       
        
        # Combine factors
        goal_probability = (
            0.5 * distance_factor + 
            0.5 * (1 - line_obstruction)
            # 0.3 * heat_map_factor
        )
        
        GLOBAL_PROBABILITIES.update({
            'goal_distance_factor': distance_factor,
            'goal_heat_map_factor': heat_map_factor,
            'goal_line_obstruction': line_obstruction,
            'final_goal_probability': goal_probability
        })

        # Store individual factors
        return max(0, min(1, goal_probability))
    

    def calculate_pass_line_obstruction(self, pos1, pos2) -> float:
        """
        Calculate pass line obstruction by opponents.
        Considers both the opponent's perpendicular distance from the pass line
        and their relative position along the pass line. Opponents closer to the 
        passer (lower t value) are considered less likely to react quickly.
        """

        # Passer and receiver positions
        x1, y1 = pos1
        x2, y2 = pos2

        # Helper function: Calculate the projection of point (x, y) onto the line from (x1, y1) to (x2, y2)
        def project_point(x, y):
            # Vector from passer to the point
            dx, dy = x - x1, y - y1
            # Vector from passer to receiver
            line_dx, line_dy = x2 - x1, y2 - y1
            line_length_sq = line_dx**2 + line_dy**2
            # Avoid division by zero
            t = (dx * line_dx + dy * line_dy) / line_length_sq if line_length_sq != 0 else 0
            return t

        # Helper function: Calculate perpendicular distance from point (x, y) to the pass line
        def point_line_distance(x, y):
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            return numerator / denominator if denominator != 0 else 0
        
        def get_individual_obstruction_factor(position: Position):
            normalize_factor = 3 #meters
            # Calculate how far along the pass line the opponent is (0 = at passer, 1 = at receiver)
            t = project_point(position.x, position.y)
            # Only consider opponents that are between the passer and receiver
            if 0 <= t <= 1:
                dist = point_line_distance(position.x, position.y)
                # Calculate the basic obstruction contribution based on distance:
                # Closer opponents (lower dist) contribute more.
                if dist<BOT_SIZE/2 + 10:
                    basic_obstruction = 9999     #  equivalent to infinite obstruction 
                basic_obstruction = max(0, 1 - dist / normalize_factor)
                # Adjust by t so that opponents closer to the passer (lower t) contribute less.
                return (t+0.2) * basic_obstruction
            return 0.0


        def bot_at_ends(botPos : Position):
            TOLERANCE = 1.5
            return (((botPos.x - x1) ** 2 + (botPos.y - y1) ** 2) < (BOT_SIZE * BOT_SIZE * TOLERANCE)) or (((botPos.x - x2) ** 2 + (botPos.y - y2) ** 2) < (BOT_SIZE * BOT_SIZE * TOLERANCE))

        obstruction_factor = 0.0
        for opp in self.opponent_robots:
            obstruction_factor += get_individual_obstruction_factor(opp.position)
        for teammate in self.our_robots:
            if bot_at_ends(teammate.position):
                continue
            obstruction_factor += get_individual_obstruction_factor(teammate.position)


        return obstruction_factor



    def calculate_pass_probability(self, bot: Robot, teammate: Robot) -> float:
        """
        Compute pass probability with multiple factors:
        1. Distance between bots
        2. Line obstruction by opponents
        3. Strategic heat map values
        4. Bot alignments
        """
        # Distance factor (inversely proportional)
        dx = bot.position.x - teammate.position.x
        dy = bot.position.y - teammate.position.y
        distance = math.sqrt(dx*dx + dy*dy)
        distance_factor = 1 - (distance / 10)  # Normalization
        
        # Line obstruction
        line_obstruction = self.calculate_pass_line_obstruction((bot.position.x,bot.position.y), (teammate.position.x,teammate.position.y))
        
        # Prefer passes forward
        forward_factor = 1.0 if bot.position.x - teammate.position.x < 0 else 0.0
        
        # Combine factors (you can adjust weights)
        pass_probability = (
            0.5 * distance_factor + 
            0.5 * (1 - line_obstruction)
        ) * forward_factor

        GLOBAL_PROBABILITIES.update({
            'pass_distance_factor': distance_factor,
            'pass_line_obstruction': line_obstruction,
            'final_pass_probability': pass_probability
        })
        
        return max(0, min(1, pass_probability))

    def calculate_future_weighted_pass_score(self, shooter: Robot, receiver: Robot, depth: int, current_probability: float, line: List[Position]) -> Tuple[float, List[Position]]:
        
        MAX_DEPTH = 3

        GOAL_PROBABILITY_THRESHOLD = 0.45
        
        GOAL_SCALING_FACTOR = 3.0
        INCOMPLETE_LINE_FACTOR = 0.8

        EXPONENTIAL_FACTOR = 1.0
        decay_factor = np.exp(-EXPONENTIAL_FACTOR * depth)

        new_line = line + [receiver.position]

        if depth > MAX_DEPTH:
            heat_map_value = self.heat_generator.goal_direction_map()[int(receiver.position.y), int(receiver.position.y)]/255.0
            return (current_probability + decay_factor * heat_map_value * INCOMPLETE_LINE_FACTOR), line

        receiver_goal_probability = self.calculate_goal_probability(receiver)
        if receiver_goal_probability > GOAL_PROBABILITY_THRESHOLD:
            return (current_probability + decay_factor * receiver_goal_probability * GOAL_SCALING_FACTOR), line + [self.field.opponent_goal]
        else:
            max_score = current_probability
            max_line = new_line
            for teammate in self.our_robots:
                if teammate.id == receiver.id:
                    continue
                pass_probability = self.calculate_pass_probability(receiver, teammate)
                if not pass_probability:
                    continue
                pass_score, pass_line = self.calculate_future_weighted_pass_score(receiver, teammate, depth+1, (decay_factor * pass_probability) + current_probability, new_line) 
                if pass_score > max_score:
                    max_score = pass_score
                    max_line = pass_line
            return max_score, max_line

    def calculate_goal_and_pass_probabilities(self) -> Tuple[float, Dict[int, float], Dict[int, List[Position]]]:
        """
        Calculates goal and pass probabilities for the current ball handler.
        Maintains the original function signature.
        """

        ball_holder_id = self.state.ball_holder
        if ball_holder_id < 0 or ball_holder_id >= len(self.our_robots):
            return 0.0, {}, {}
        
        shooter = self.our_robots[ball_holder_id]
        
        # Calculate goal probability with detailed factors
        goal_prob = self.calculate_goal_probability(shooter)
        # GLOBAL_PROBABILITIES['final_goal_probabilities'].append(goal_prob)
        
        # Calculate pass probabilities
        pass_probs = {}
        pass_lines = {}
        scale = 0.0
        for teammate in self.our_robots:
            if teammate.id != shooter.id:
                pass_prob = self.calculate_pass_probability(shooter, teammate)

                if pass_prob > scale:
                    scale = pass_prob

                pass_probs[teammate.id], pass_lines[teammate.id] = self.calculate_future_weighted_pass_score(shooter, teammate, 0, pass_prob, [shooter.position])

        scale = scale / max(pass_probs.values())
        for key in pass_probs.keys():
            pass_probs[key] = pass_probs[key] * scale
        
        return goal_prob, pass_probs, pass_lines

    def draw_pass_line(self, line: List[Position], image):
        if(line == None):
            return
        scale_factor = self.visualizer.scale
        sy, sx = self.visualizer.get_nearest_index((line[0].x, line[0].y))
        sy = (int)(sy * scale_factor)
        sx = (int)(sx * scale_factor)
        for i in range(1, len(line)):
            ey, ex = self.visualizer.get_nearest_index((line[i].x, line[i].y))
            ey = (int)(ey * scale_factor)
            ex = (int)(ex * scale_factor)
            red = (int)((255*(i-1))/(len(line)-2)) if len(line) != 2 else 0
            self.visualizer.draw_dotted_line(image, (sx, sy), (ex, ey), (0, 255, red))
            sx, sy = ex, ey

    # --- Modified "We Have Ball" logic ---
    def _handle_we_have_ball(self):
        """Handle decision making when we have the ball—with shot/pass logic."""
        self.get_logger().info("\nHandling WE_HAVE_BALL state:")
        ball_holder_id = self.state.ball_holder
        if ball_holder_id < 0 or ball_holder_id >= len(self.our_robots):
            self.get_logger().warn("Invalid ball holder ID in WE_HAVE_BALL state")
            return
        ball_handler = self.our_robots[ball_holder_id]
        
        # Record where the ball was taken if not already set.
        if ball_handler.possession_position is None:
            ball_handler.possession_position = Position(ball_handler.position.x,
                                                        ball_handler.position.y,
                                                        ball_handler.position.z)
        
        # Calculate probabilities.
        goal_prob, pass_probs, pass_lines = self.calculate_goal_and_pass_probabilities()
        best_pass_id = None
        best_pass_prob = 0.0
        for teammate_id, prob in pass_probs.items():
            if prob > best_pass_prob:
                best_pass_prob = prob
                best_pass_id = teammate_id
        
        self.get_logger().info(f"Pass_prob = {best_pass_prob} to bot{best_pass_id} , Goal_prob = {goal_prob}")

        # Always plan to advance toward the opponent goal.
        goal_pos = self.field.opponent_goal
        dx = goal_pos.x - ball_handler.possession_position.x
        dy = goal_pos.y - ball_handler.possession_position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        max_move = 3.0  # rule: bot cannot move more than 3 meters away from possession point.
        scale = max_move / distance_to_goal if distance_to_goal > max_move else 1.0
        target_x = ball_handler.possession_position.x + dx * scale
        target_y = ball_handler.possession_position.y + dy * scale
        
        ball_handler.target_position = Position(target_x, target_y, ball_handler.position.z)
        ball_handler.target_theta = math.atan2(goal_pos.y - ball_handler.position.y,
                                               goal_pos.x - ball_handler.position.x)
        self.get_logger().info(f"Ball handler {ball_handler.id} advancing toward goal at ({target_x:.2f}, {target_y:.2f}) (goal_prob: {goal_prob:.2f})")
        
        # Decision branches:
        GOAL_PROB_THRESHOLD = 0.45
        PASS_PROB_THRESHOLD = 0.7
        
        if goal_prob > GOAL_PROB_THRESHOLD:
            # If near the target, attempt a shot.
            pos_error = math.sqrt((ball_handler.position.x - target_x)**2 + (ball_handler.position.y - target_y)**2)
            if pos_error < 0.5:
                command_msg = String()
                kick_speed = 700 # For a shot, use a relatively high speed.
                command_msg.data = f"KICK {int(kick_speed)} 45"
                self.command_pub.publish(command_msg)
                self.get_logger().info(f"Ball handler {ball_handler.id} shooting with command: {command_msg.data}")
                
        elif best_pass_id is not None and best_pass_prob > PASS_PROB_THRESHOLD:
            # Attempt a pass if one is not already in progress.
            if not self.pass_in_progress:
                receiver = self.our_robots[best_pass_id]
                # Stop both bots by setting their target positions to their current positions.
                ball_handler.target_position = Position(ball_handler.position.x,
                                                        ball_handler.position.y,
                                                        ball_handler.position.z)
                receiver.target_position = Position(receiver.position.x,
                                                      receiver.position.y,
                                                      receiver.position.z)
                # Orient the two bots toward each other.
                angle_handler_to_receiver = math.atan2(receiver.position.y - ball_handler.position.y,
                                                       receiver.position.x - ball_handler.position.x)
                angle_receiver_to_handler = math.atan2(ball_handler.position.y - receiver.position.y,
                                                       ball_handler.position.x - receiver.position.x)
                ball_handler.target_theta = angle_handler_to_receiver
                receiver.target_theta = angle_receiver_to_handler
                self.get_logger().info(f"Initiating pass from {ball_handler.id} to {receiver.id} (pass_prob: {best_pass_prob:.2f})")
                
                status_msg = String()
                status_msg.data = f"PASS_INITIATED from {ball_handler.id} to {receiver.id}"
                self.pass_status_pub.publish(status_msg)
                # Compute the distance between ball handler and receiver.
                d = math.sqrt((ball_handler.position.x - receiver.position.x)**2 +
                              (ball_handler.position.y - receiver.position.y)**2)
                # For the pass, choose a speed proportional to the distance.
                base_speed = 200
                factor = 50
                pass_speed = base_speed + factor * d
                # Compute the expected transit time.
                self.pass_timeout = d / pass_speed if pass_speed > 0 else 2.0

                if (abs(ball_handler.current_theta - ball_handler.target_theta) < PASS_ALIGNMENT_THRESHOLD) and (abs(receiver.current_theta - receiver.target_theta) < PASS_ALIGNMENT_THRESHOLD):
                # Now it is safe to execute the pass.
                    command_msg = String()
                    command_msg.data = f"KICK {int(pass_speed)} 0"
                    self.command_pub.publish(command_msg)
                    self.get_logger().info(f"Ball handler {ball_handler.id} passing with command: {command_msg.data}")
                    status_msg.data = f"PASS_EXECUTED from {ball_handler.id} to {receiver.id}"
                    self.pass_status_pub.publish(status_msg)
                    # Mark that a pass is now underway. (No robot has possession.)
                    self.pass_in_progress = True
                    self.pass_receiver_id = receiver.id
                    self.pass_start_time = time.time()
                else:
                    # Optionally log or wait until alignment is achieved.
                    self.get_logger().info("Waiting for proper alignment before passing...")

            else:
                self.get_logger().info("Pass already in progress, waiting for receiver to gain possession.")
        else:
            # Neither opportunity is strong—still advance toward the goal.
            self.get_logger().info(f"Ball handler {ball_handler.id} advancing as default action.")
        
        # Assign strategic positions for supporting robots.
        # Exclude the ball handler and, if a pass is underway, also exclude the intended receiver.
        excluded_ids = {ball_handler.id}
        if self.pass_in_progress:
            excluded_ids.add(self.pass_receiver_id)
        available_robots = [robot for robot in self.our_robots if robot.id not in excluded_ids]
        strategic_positions, image = self.generate_strategic_positions()[:len(available_robots)]
        
        self.draw_assignment_lines(image)
        if best_pass_id is not None:
            self.draw_pass_line(pass_lines[best_pass_id], image)
        self.draw_heatmap(image)
        
        assignments = self.assign_positions(strategic_positions, available_robots)
        for robot_id, target_pos in assignments.items():
            self.our_robots[robot_id].target_position = target_pos
            # self.get_logger().info(f"Robot {robot_id} moving to support at position ({target_pos.x:.2f}, {target_pos.y:.2f})")

    def _handle_loose_ball(self):
        """Handle decision making when the ball is loose (only invoked when no pass is underway)."""
        if self.pass_in_progress:
            self.get_logger().info("Pass in progress: loose ball routine skipped.")
            return

        self.get_logger().info("\nHandling LOOSE_BALL state:")
        # Find the closest robot to the ball.
        closest_robot = min(self.our_robots, 
                            key=lambda r: np.sqrt((r.position.x - self.ball.position.x)**2 + 
                                                  (r.position.y - self.ball.position.y)**2))
        closest_robot.target_position = self.ball.position

        dx = self.ball.position.x - closest_robot.position.x
        dy = self.ball.position.y - closest_robot.position.y
        distance = sqrt(dx**2 + dy**2)
        if distance < 1e-6 or distance < BOT_SIZE / 2:
            target_x, target_y = self.ball.position.x, self.ball.position.y
        else:
            target_x = self.ball.position.x - (BOT_SIZE / 2) * (dx / distance)
            target_y = self.ball.position.y - (BOT_SIZE / 2) * (dy / distance)
        target_theta = atan2(target_y - closest_robot.position.y, target_x - closest_robot.position.x)
        closest_robot.target_theta = target_theta

        self.get_logger().info(f"Robot {closest_robot.id} is closest to ball - moving to get possession")
        self.state.ball_holder = closest_robot.id
        closest_robot.possession_position = Position(closest_robot.position.x,
                                                      closest_robot.position.y,
                                                      closest_robot.position.z)
        # Generate positions for other robots.
        available_robots = [robot for robot in self.our_robots if robot.id != closest_robot.id]
        strategic_positions, image = self.generate_strategic_positions()
        
        self.draw_assignment_lines(image)
        self.draw_heatmap(image)

        assignments = self.assign_positions(strategic_positions, available_robots)
        for robot_id, target_pos in assignments.items():
            self.our_robots[robot_id].target_position = target_pos
            self.get_logger().info(f"Robot {robot_id} moving to support at position ({target_pos.x:.2f}, {target_pos.y:.2f})")

    def _handle_opponent_has_ball(self):
        """Basic handling when an opponent has the ball."""
        self.get_logger().info("\nHandling OPPONENT_HAS_BALL state:")
        strategic_positions, image = self.generate_strategic_positions()
        self.draw_assignment_lines(image)
        self.draw_heatmap(image)
        # assignments = self.assign_positions(strategic_positions, available_robots)

        closest_opp = min(self.opponent_robots, 
                          key=lambda r: np.sqrt((r.position.x - self.ball.position.x)**2 + 
                                                (r.position.y - self.ball.position.y)**2))
        closest_our = min(self.our_robots, 
                          key=lambda r: np.sqrt((r.position.x - self.ball.position.x)**2 + 
                                                (r.position.y - self.ball.position.y)**2))
        blocking_pos = Position(
            closest_opp.position.x - 1.5,
            closest_opp.position.y,
            0.0
        )
        closest_our.target_position = blocking_pos
        closest_our.target_theta = math.atan2(
            self.ball.position.y - blocking_pos.y,
            self.ball.position.x - blocking_pos.x
        )
        available_robots = [robot for robot in self.our_robots 
                            if robot.id != closest_our.id and robot.role != RobotRole.GOALKEEPER]
        defensive_positions = []
        base_x = (self.field.our_goal.x + self.ball.position.x) / 2
        spread = 2.0
        for i in range(len(available_robots)):
            y_pos = self.ball.position.y + spread * (i - len(available_robots)/2)
            defensive_positions.append(Position(base_x, y_pos, 0.0))
        assignments = self.assign_positions(defensive_positions, available_robots)
        for robot_id, target_pos in assignments.items():
            self.our_robots[robot_id].target_position = target_pos
            self.our_robots[robot_id].target_theta = math.atan2(
                self.ball.position.y - target_pos.y,
                self.ball.position.x - target_pos.x
            )
            self.get_logger().info(
                f"Robot {robot_id} defending at position ({target_pos.x:.2f}, {target_pos.y:.2f})"
            )

    def generate_strategic_positions(self) -> Tuple[List[Position], np.ndarray]:
        """Generate strategic positions using heatmaps based on game state."""
        maps = []
        weights = []
        maps.extend([
            self.heat_generator.robots_repulsion_map(),
            self.heat_generator.vertical_center_attraction_map(),
            self.heat_generator.horizontal_right_attraction_map(),
        ])
        weights.extend([0.6, 0.15, 0.15,0.2])
        if self.game_state == GameState.WE_HAVE_BALL:
            maps.extend([
                self.heat_generator.ideal_pass_distance_map(),
                self.heat_generator.goal_direction_map()
            ])
            weights.extend([0.2, 0.15])
        elif self.game_state == GameState.LOOSE_BALL:
            maps.extend([
                self.heat_generator.ideal_pass_distance_map(),
                self.heat_generator.goal_direction_map()
            ])
            weights.extend([0.2, 0.15])
        elif self.game_state == GameState.OPPONENT_HAS_BALL:
            maps.extend([
                self.heat_generator.defensive_opponent_influence_map(),
                self.heat_generator.goal_direction_map()
            ])
            weights.extend([0.2, 0.15])
        combined_map = self.heat_generator.combine_heat_maps(maps, weights)
        positions = self.clusterer.get_strategic_positions(combined_map)
        image = self.visualizer.get_opencv_visualization_aligned(combined_map, positions)
        return [Position(pos[0], pos[1]) for pos in positions], image    
    
    def draw_heatmap(self, image):
        image = cv2.flip(image, 0)
        cv2.imshow("heatmap", image)
        cv2.waitKey(1)

    def assign_positions(self, strategic_positions: List[Position], robots: List[Robot]) -> Dict[int, Position]:
        """Assign robots to positions using the Hungarian algorithm while preserving robot IDs."""
        cost_matrix = np.zeros((len(robots), len(strategic_positions)))
        for i, robot in enumerate(robots):
            for j, pos in enumerate(strategic_positions):
                cost_matrix[i, j] = np.sqrt((robot.position.x - pos.x)**2 +
                                            (robot.position.y - pos.y)**2)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        assignments = {}
        for i, j in zip(row_indices, col_indices):
            real_robot_id = robots[i].id
            assignments[real_robot_id] = strategic_positions[j]
        return assignments

    def check_pass_status(self):
        """Check whether the intended receiver has gained possession within the allowed time."""
        if self.pass_in_progress:
            # Instead of checking ball-to-robot distance, we use state.ball_holder.
            if self.state.ball_holder == self.pass_receiver_id:
                self.get_logger().info(f"Pass successful to robot {self.pass_receiver_id}")
                status_msg = String()
                status_msg.data = f"PASS_SUCCESS from pass to {self.pass_receiver_id}"
                self.pass_status_pub.publish(status_msg)
                self.pass_in_progress = False
            elif time.time() - self.pass_start_time > self.pass_timeout:
                self.get_logger().warn(f"Pass from robot {self.state.ball_holder} to robot {self.pass_receiver_id} failed")
                status_msg = String()
                status_msg.data = f"PASS_FAILED from {self.state.ball_holder} to {self.pass_receiver_id}"
                self.pass_status_pub.publish(status_msg)
                self.pass_in_progress = False

    def execute_decision_making(self):
        """Main decision-making loop (logic now invoked in decision_making_callback)."""
        pass  # Not used because decision_making_callback already wraps the logic.

def main(args=None):
    rclpy.init(args=args)
    state = RoboCupState()
    generator = HeatMapGenerator(state)
    clusterer = HeatMapClusterer(state)
    visualizer = HeatMapVisualizer(state)
    game_manager = GameManagerROS2(state, generator, clusterer, visualizer)
    try:
        rclpy.spin(game_manager)
    except KeyboardInterrupt:
        pass
    finally:
        game_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
