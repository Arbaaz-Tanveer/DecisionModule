import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import time 
import os
import math

class RoboCupState:
    def __init__(self):
        # Field dimensions (in meters)
        self.field_length = 22.0
        self.field_width = 14.0

        # Initialize with example positions
        self.our_positions = np.array([
            [-2.0, 1.0],   # Player 1 (ball holder)
            [-6.0, -1.5],  # Player 2
            [4.0, 0.0],    # Player 3
            [1.0, 5.0],    # Player 4
            [2.0, -2.0]    # Player 5
        ])
        
        self.opp_positions = np.array([
            [5.0, 5.0],    # Opponent 1
            [1.0, -1.0],   # Opponent 2
            [0.0, 1.0],    # Opponent 3
            [-1.0, -2.0],  # Opponent 4
            [-2.0, 2.0]    # Opponent 5
        ])
        
        self.our_velocities = np.array([
            [0.5, 0.0],    # Player 1 velocity
            [0.0, 0.5],    # Player 2 velocity
            [-0.5, 0.0],   # Player 3 velocity
            [0.0, -0.5],   # Player 4 velocity
            [0.3, 0.3]     # Player 5 velocity
        ])
        
        self.opp_velocities = np.array([
            [-0.3, 0.0],   # Opponent 1 velocity
            [0.0, -0.3],   # Opponent 2 velocity
            [0.3, 0.0],    # Opponent 3 velocity
            [0.0, 0.3],    # Opponent 4 velocity
            [-0.2, -0.2]   # Opponent 5 velocity
        ])
        
        self.ball_holder = 0  # Index of our robot holding the ball
        
        self.ball_position = np.array([0,0])
        # Resolution for heat maps (pixels per meter)
        self.resolution = 10
        
        # Initialize grid
        self.x = np.linspace(-self.field_length/2, self.field_length/2, 
                             int(self.field_length * self.resolution))
        self.y = np.linspace(-self.field_width/2, self.field_width/2, 
                             int(self.field_width * self.resolution))
        self.X, self.Y = np.meshgrid(self.x, self.y)

class HeatMapGenerator:
    def __init__(self, state: RoboCupState):
        self.state = state
        
    def get_distance_from_point(self, point):
        """Calculate distance from each grid point to a specific point"""
        return np.sqrt((self.state.X - point[0])**2 + (self.state.Y - point[1])**2)
    
    def robots_repulsion_map(self, sigma=1.0):
        """Generate heat map where values increase away from all robots"""
        heat_map = np.zeros_like(self.state.X)
        
        # Add repulsion from our robots
        for pos in self.state.our_positions:
            distance = self.get_distance_from_point(pos)
            heat_map += 1 - np.exp(-distance**2 / (2*sigma**2))
            
        # Add repulsion from opponent robots
        for pos in self.state.opp_positions:
            distance = self.get_distance_from_point(pos)
            heat_map += 1 - np.exp(-distance**2 / (2*sigma**2))
            
        return heat_map / heat_map.max()  # Normalize
    
    def vertical_center_attraction_map(self):
        """Generate heat map with higher values near vertical center"""
        return 1 - np.abs(self.state.Y) / (self.state.field_width/2)
    
    def horizontal_right_attraction_map(self):
        # Create gradient using X coordinates
        # Normalize X coordinates to [0,1] range where:
        # leftmost = 0.0, rightmost = 1.0
        x_normalized = (self.state.X - self.state.X.min()) / (self.state.X.max() - self.state.X.min())
        return x_normalized
    
    def ball_holder_circle_map(self, radius=1.5):
        """Generate circular region around ball holder"""
        heat_map = np.zeros_like(self.state.X)
        
        # Create circles around all our robots with the ball holder having higher value
        for i, pos in enumerate(self.state.our_positions):
            distance = self.get_distance_from_point(pos)
            if i == self.state.ball_holder:
                heat_map += (distance <= radius).astype(float) * 1.0  # Full intensity for ball holder
                 
        return heat_map / heat_map.max()
    
    def ideal_pass_distance_map(self, A=1.0, r0=3.0, sigma=1.0):
        """Generate heat map based on ideal pass distance equation"""
        # Calculate pass distance map from ball holder
        holder_pos = self.state.our_positions[self.state.ball_holder]
        r = self.get_distance_from_point(holder_pos)
        heat_map = A * np.exp(-(r - r0)**2 / (2*sigma**2))
            
        return heat_map / heat_map.max()
    
    def goal_direction_map(self, goal_pos=(10.2, 0.0), IGD=6.0, sigma=1.0, p=1.0):
        """
        Generate heat map based on the goal probability equation (no dependence on a ball-holder):

        GoalProb = cos(α) * (p / (dist * sqrt(2π))) * exp( - (dist - IGD)^2 / (2*sigma^2) )

        where:
        - dist is distance from each grid cell to the goal
        - α is the angle (w.r.t. some fixed axis) from each grid cell toward the goal
        - IGD is the ideal goal distance
        - sigma is the std dev in the Gaussian
        - p is a scaling parameter
        """

        # dx, dy: vectors from each grid point to the goal
        dx = goal_pos[0] - self.state.X
        dy = goal_pos[1] - self.state.Y

        # Angle of each grid point w.r.t. the goal (α)
        angles = np.arctan2(dy, dx)
        cos_alpha = np.cos(angles)

        # Distance from each grid point to the goal
        dist_to_goal = np.sqrt(dx**2 + dy**2)

        # Gaussian component centered at IGD
        gaussian = np.exp(-((dist_to_goal - IGD)**2) / (2 * sigma**2))

        # Normalization factor = p / (dist * sqrt(2π))
        # (Prevent divide-by-zero by clipping distance)
        dist_clipped = np.clip(dist_to_goal, 1e-6, None)
        norm_factor = p / (dist_clipped * np.sqrt(2 * np.pi))

        # Combine the components
        heat_map = cos_alpha * norm_factor * gaussian

        # Clip negative values and rescale to [0, 1]
        heat_map = np.clip(heat_map, 0, None)
        if heat_map.max() > 0:
            heat_map /= heat_map.max()

        return heat_map

    
    def goalpost_entrance_map(self, x_min=8.0, y_min=-3.0, y_max=3.0):
        """
        Generate a heat map that highlights a rectangular region at the right portion of the field,
        representing the area inside the goalpost.
        
        Parameters:
        - x_min: Minimum x-coordinate (left boundary of rectangle)
        - y_min: Minimum y-coordinate (bottom boundary)
        - y_max: Maximum y-coordinate (top boundary)
        """
        # Create a binary map: 1 inside the rectangle, 0 outside
        heat_map = np.ones_like(self.state.X)
        # Field boundaries (assuming symmetric field with center at 0)
        x_max = self.state.field_length / 2
        mask = (self.state.X >= x_min) & (self.state.X <= x_max) & \
               (self.state.Y >= y_min) & (self.state.Y <= y_max)
        heat_map[mask] = 0.0
        return heat_map  # Already binary; normalization unnecessary
    
    def defensive_opponent_influence_map(self, r_max=7.0):
        """
        Generate a heat map based on defensive influence from opponents.
        For each opponent, the influence at a point is:
            p_influence = cos(α) * (r_max - d)
        where:
         - α is the angle between the vector from opponent to ball and the vector from opponent to the grid point.
         - d is the distance from the opponent to the grid point.
         - Only points with d < r_max contribute.
        The influences from all opponents are summed.
        
        This map suggests positions that are effective in blocking an opponent while
        keeping the ball in sight.
        """
        heat_map = np.zeros_like(self.state.X)
        ball_pos = self.state.ball_position
        
        # Iterate through each opponent
        for opp in self.state.opp_positions:
            # Vector from opponent to ball
            vec_ball = np.array([ball_pos[0] - opp[0], ball_pos[1] - opp[1]])
            norm_vec_ball = np.linalg.norm(vec_ball)
            # Avoid division by zero; if an opponent is exactly at the ball, skip
            if norm_vec_ball == 0:
                continue
            
            # Compute difference arrays relative to opponent's position
            diff_x = self.state.X - opp[0]
            diff_y = self.state.Y - opp[1]
            distance = np.sqrt(diff_x**2 + diff_y**2)
            
            # Calculate cosine of the angle between (opp -> ball) and (opp -> grid point)
            # For each grid point, dot product divided by (norm(vec_ball) * distance)
            dot_product = diff_x * vec_ball[0] + diff_y * vec_ball[1]
            # To avoid division by zero when distance==0, set cosine = 1 there
            cos_alpha = np.where(distance > 0, dot_product / (norm_vec_ball * distance), 1.0)
            
            # Only consider grid points within the maximum radius
            influence = np.where(distance < r_max, cos_alpha * (r_max - distance), 0)
            # Sum the influence of all opponents
            heat_map += influence
        
        # Normalize the heat map if non-zero
        if heat_map.max() > 0:
            heat_map = heat_map / heat_map.max()
        return heat_map
    
    def combine_heat_maps(self, maps, weights=None):
        """Combine multiple heat maps with optional weights"""
        if weights is None:
            weights = [1.0] * len(maps)
        
        combined = np.zeros_like(self.state.X)
        for map_data, weight in zip(maps, weights):
            combined += weight * map_data
            
        # Normalize if the sum is non-zero
        if combined.max() > 0:
            combined = combined / combined.max()
        return combined

class HeatMapClusterer:
    def __init__(self, state: RoboCupState):
        self.state = state
        
    def find_optimal_positions(self, heat_map, n_clusters=5, max_points = None):
        """Find optimal positions in the heatmap using clustering"""
        high_value_points = np.argwhere(heat_map > 0.85)
        
        if len(high_value_points) < n_clusters:
            return np.array([[0, 0]] * n_clusters)
        
        if max_points is not None and len(high_value_points) > max_points:
            indices = np.random.choice(len(high_value_points), max_points, replace=False)
            high_value_points = high_value_points[indices]

        weights = np.array([heat_map[x, y] for x, y in high_value_points])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(high_value_points, sample_weight=weights)
        
        centers = kmeans.cluster_centers_
        x_coords = self.state.x[np.clip(centers[:, 1].astype(int), 0, len(self.state.x)-1)]
        y_coords = self.state.y[np.clip(centers[:, 0].astype(int), 0, len(self.state.y)-1)]
        
        return np.column_stack((x_coords, y_coords))
    
    def get_nearest_index(self, position):
        """Convert field coordinates to heatmap indices"""
        x_idx = np.argmin(np.abs(self.state.x - position[0]))
        y_idx = np.argmin(np.abs(self.state.y - position[1]))
        return y_idx, x_idx
    
    def get_strategic_positions(self, combined_map):
        """Get strategic positions based on the combined heatmap"""
        positions = self.find_optimal_positions(combined_map)
        position_values = [combined_map[self.get_nearest_index(pos)] for pos in positions]
        return positions[np.argsort(position_values)[::-1]]
    

        
class HeatMapVisualizer:
    def __init__(self, state: RoboCupState):
        self.state = state
        self.scale = 4

    def show_matplotlib(self, heat_map, title="Heat Map"):
        """Display heat map using matplotlib with blue-red colormap"""
        plt.figure(figsize=(10, 8))
        
        # Create custom colormap from blue to red
        cmap = plt.cm.RdBu_r  # Built-in blue-red colormap
        
        # Plot heatmap
        plt.imshow(heat_map, extent=[-self.state.field_length/2, self.state.field_length/2,
                                   -self.state.field_width/2, self.state.field_width/2],
                  origin='lower', cmap=cmap)
        
        # Plot robot positions
        plt.plot(self.state.our_positions[:, 0], self.state.our_positions[:, 1], 
                'go', label='Our Team', markersize=10)
        plt.plot(self.state.opp_positions[:, 0], self.state.opp_positions[:, 1], 
                'yo', label='Opponents', markersize=10)
        
        # Highlight ball holder
        plt.plot(self.state.our_positions[self.state.ball_holder, 0],
                self.state.our_positions[self.state.ball_holder, 1],
                'ro', markersize=15, label='Ball Holder')
        
        plt.colorbar(label='Value')
        plt.title(title)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_nearest_index(self, position):
        """Convert field coordinates to heatmap indices"""
        x_idx = np.argmin(np.abs(self.state.x - position[0]))
        y_idx = np.argmin(np.abs(self.state.y - position[1]))
        return y_idx, x_idx
    
    def visualize_clusters(self, heat_map, positions, title="Clustered Strategic Positions"):
        """Visualize heatmap with clustered positions"""
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        plt.imshow(heat_map, extent=[-self.state.field_length/2, self.state.field_length/2,
                                   -self.state.field_width/2, self.state.field_width/2],
                  origin='lower', cmap='RdBu_r')
        
        # Plot existing robots
        plt.plot(self.state.our_positions[:, 0], self.state.our_positions[:, 1], 
                'go', label='Our Team', markersize=10)
        plt.plot(self.state.opp_positions[:, 0], self.state.opp_positions[:, 1], 
                'yo', label='Opponents', markersize=10)
        
        # Plot ball holder
        plt.plot(self.state.our_positions[self.state.ball_holder, 0],
                self.state.our_positions[self.state.ball_holder, 1],
                'ro', markersize=15, label='Ball Holder')
        
        # Plot cluster positions
        for i, pos in enumerate(positions):
            plt.plot(pos[0], pos[1], 'w*', markersize=15, 
                    label=f'Strategic Position {i+1}' if i==0 else "")
            plt.annotate(f'P{i+1}', (pos[0], pos[1]), 
                        xytext=(10, 10), textcoords='offset points',
                        color='white', fontsize=12)
        
        plt.colorbar(label='Heat Value')
        plt.title(title)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def get_opencv_visualization(self, heat_map, positions):
        """Get OpenCV image with clusters"""
        heat_map_normalized = (heat_map * 255).astype(np.uint8)
        heat_map_color = cv2.applyColorMap(heat_map_normalized, cv2.COLORMAP_JET)
        
        # Convert heatmap coordinates to image coordinates
        for pos in positions:
            x, y = self.get_nearest_index(pos)
            cv2.drawMarker(heat_map_color, (y, x), (255, 255, 255), 
                          cv2.MARKER_STAR, 20, 2)
            
        return heat_map_color
    
    def draw_dotted_line(self, img, start_pt, end_pt, color, thickness=2, gap=10):
        """
        Draw a dotted line on img from start_pt to end_pt (both are (x, y) in image coords).
        gap sets the length of spacing between dots.
        """
        dist_x = end_pt[0] - start_pt[0]
        dist_y = end_pt[1] - start_pt[1]
        length = int((dist_x**2 + dist_y**2)**0.5)
        
        # Avoid division by zero if start and end are the same
        if length == 0:
            return
        
        step_x = dist_x / length
        step_y = dist_y / length
        
        # Plot small circles along the line
        for i in range(0, length, gap):
            cx = int(start_pt[0] + step_x * i)
            cy = int(start_pt[1] + step_y * i)
            cv2.circle(img, (cx, cy), thickness, color, -1)

    def get_opencv_visualization_aligned(self, heat_map, positions, title="Clustered Positions"):
        """
        Create an OpenCV visualization with a colormap aligned to Matplotlib's RdBu_r.
        """
        # Normalize the heatmap to [0, 1]
        heat_map_normalized = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())

        # Apply Matplotlib colormap (RdBu_r)
        cmap = plt.cm.RdBu_r  # Matplotlib's colormap
        mapped_colors = (cmap(heat_map_normalized)[:, :, :3] * 255).astype(np.uint8)  # RGB values

        # Convert RGB to BGR for OpenCV
        heat_map_bgr = cv2.cvtColor(mapped_colors, cv2.COLOR_RGB2BGR)
        # Suppose you want it scaled by a factor of 2
        scale_factor = self.scale
        new_width = int(heat_map_bgr.shape[1] * scale_factor)
        new_height = int(heat_map_bgr.shape[0] * scale_factor)
        heat_map_bgr = cv2.resize(heat_map_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Example: draw our team with green circles
        for pos in self.state.our_positions:
            x_idx, y_idx = self.get_nearest_index(pos)
            # Scale indexes if you've resized the heat_map_bgr
            x_idx = int(x_idx * scale_factor)
            y_idx = int(y_idx * scale_factor)
            cv2.circle(heat_map_bgr, (y_idx, x_idx), 8, (0, 255, 0), -1)

        # Example: draw opponent team with yellow circles
        for pos in self.state.opp_positions:
            x_idx, y_idx = self.get_nearest_index(pos)
            x_idx = int(x_idx * scale_factor)
            y_idx = int(y_idx * scale_factor)
            cv2.circle(heat_map_bgr, (y_idx, x_idx), 8, (0, 255, 255), -1)
        # Overlay cluster positions

        for pos in positions:
            x_idx, y_idx = self.get_nearest_index(pos)
            cv2.drawMarker(heat_map_bgr, (int(y_idx*scale_factor), int(x_idx*scale_factor)), (255, 255, 255), cv2.MARKER_STAR, 20, 2)
       
        return heat_map_bgr

# Example usage:
if __name__ == "__main__":
    # Initialize state
    state = RoboCupState()
    
    # Create generators and visualizer
    generator = HeatMapGenerator(state)
    visualizer = HeatMapVisualizer(state)
    clusterer = HeatMapClusterer(state)
    
    # Generate individual heat maps
    i_time = time.time()
    repulsion_map = generator.robots_repulsion_map()
    vertical_map = generator.vertical_center_attraction_map()
    horizontal_map = generator.horizontal_right_attraction_map()  # New map
    pass_distance_map = generator.ideal_pass_distance_map()
    goal_map = generator.goal_direction_map()
    goalpost_map = generator.goalpost_entrance_map()  # New goalpost entrance map
    defensive_map = generator.defensive_opponent_influence_map()  # New defensive opponent influence map
    
    # Combine maps (adjust weights as needed)
    combine_time = time.time()
    combined_map = generator.combine_heat_maps(
        [repulsion_map, vertical_map, horizontal_map, pass_distance_map, goal_map, goalpost_map, defensive_map],
        weights=[0.6, 0.15, 0.15, 0.2, 0.15, 0.3, 0.3]  # Adjust weights to include the new maps
    )
    print(f'Combining time = {time.time()-combine_time}')
    cluster_time = time.time()
    positions = clusterer.get_strategic_positions(combined_map)
    print(f"clustering time = {time.time() - cluster_time}")
    print("Strategic positions:\n", positions)
    
    # Visualize individual maps
    visualizer.show_matplotlib(repulsion_map, "Robots Repulsion Map")
    visualizer.show_matplotlib(vertical_map, "Vertical Center Attraction Map")
    visualizer.show_matplotlib(horizontal_map, "Horizontal Right Attraction Map")
    visualizer.show_matplotlib(pass_distance_map, "Ideal Pass Distance Map")
    visualizer.show_matplotlib(goal_map, "Goal Direction Map")
    visualizer.show_matplotlib(goalpost_map, "Goalpost Entrance Map")
    visualizer.show_matplotlib(defensive_map, "Defensive Opponent Influence Map")
    visualizer.show_matplotlib(combined_map, "Combined Heat Map")

    visualizer.visualize_clusters(combined_map, positions)
    
    # Optionally, create an OpenCV visualization
    # cv_image = visualizer.get_opencv_visualization_aligned(combined_map, positions)
    # cv2.imshow('Heat Map (OpenCV)', cv_image)
    # cv2.waitKey(0)
