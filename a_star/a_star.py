'''
The code to connect to the colab and drive
!pip install rasterio numpy matplotlib torch
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
'''
import numpy as np
import torch
import rasterio
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

class TerrainRLPathfinder:
    def __init__(self, dem_file_path: str, reduction_factor: float = 0.5):
        self.dem_file_path = dem_file_path
        self.reduction_factor = reduction_factor
        self.elevation_data = None
        self.slope_x_data = None
        self.slope_y_data = None
        self.obstacle_data = None
        self.ruggedness_data = None
        self.twi_data = None
        self.flow_accumulation = None
        self.cell_size = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_table = None
        self.episode_rewards = []

    def preprocess_dem(self):
        """Preprocess DEM by reducing pixels and calculating all terrain factors"""
        with rasterio.open(self.dem_file_path) as dem_dataset:
            original_data = dem_dataset.read(1)
            self.cell_size = dem_dataset.res[0]

        # Reduce pixels using Gaussian filter
        reduced_data = gaussian_filter(original_data, sigma=1/self.reduction_factor)
        reduced_data = reduced_data[::int(1/self.reduction_factor), ::int(1/self.reduction_factor)]

        # Convert to tensor and move to device
        self.elevation_data = torch.tensor(reduced_data, device=self.device)

        print(f"Original shape: {original_data.shape}")
        print(f"Reduced shape: {self.elevation_data.shape}")

        # Calculate all terrain factors
        self._calculate_all_factors()

    def _calculate_all_factors(self):
        """Calculate all terrain factors"""
        print("Calculating terrain factors...")

        # Calculate slopes
        self.slope_x_data, self.slope_y_data = self._calculate_slope()

        # Calculate ruggedness
        self.ruggedness_data = self._calculate_ruggedness()

        # Calculate flow accumulation (simplified)
        self.flow_accumulation = self._calculate_flow_accumulation()

        # Calculate TWI
        self.twi_data = self._calculate_twi()

        # Detect obstacles
        self.obstacle_data = self._detect_obstacles()

    def _calculate_slope(self) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_y, grad_x = torch.gradient(self.elevation_data.float())
        return grad_x / self.cell_size, grad_y / self.cell_size

    def _calculate_ruggedness(self) -> torch.Tensor:
        return torch.sqrt(self.slope_x_data**2 + self.slope_y_data**2)

    def _calculate_flow_accumulation(self) -> torch.Tensor:
        # Simplified flow accumulation calculation
        flow_acc = torch.zeros_like(self.elevation_data)
        rows, cols = self.elevation_data.shape

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Count lower neighbors as contributing flow
                neighbors = self.elevation_data[i-1:i+2, j-1:j+2]
                flow_acc[i, j] = torch.sum(neighbors > self.elevation_data[i, j]).float()

        return flow_acc

    def _calculate_twi(self) -> torch.Tensor:
        slope_magnitude = torch.sqrt(self.slope_x_data**2 + self.slope_y_data**2)
        # Avoid division by zero
        slope_magnitude = torch.where(slope_magnitude == 0, torch.tensor(0.0001, device=self.device), slope_magnitude)
        return torch.log(self.flow_accumulation / slope_magnitude)

    def _detect_obstacles(self, threshold: float = 5.0) -> torch.Tensor:
        return torch.where(
            (torch.abs(self.slope_x_data) > threshold) |
            (torch.abs(self.slope_y_data) > threshold),
            torch.ones_like(self.elevation_data),
            torch.zeros_like(self.elevation_data)
        )

    def initialize_q_table(self):
        """Initialize Q-table with zeros"""
        state_space = self.elevation_data.shape[0] * self.elevation_data.shape[1]
        action_space = 8  # 8 possible directions of movement
        self.q_table = torch.zeros((state_space, action_space), device=self.device)

    def get_state(self, position: Tuple[int, int]) -> int:
        """Convert 2D position to 1D state"""
        return position[0] * self.elevation_data.shape[1] + position[1]

    def _get_next_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next position based on current position and action"""
        row, col = position

        # Define movement based on action (0 to 7 directions: N, NE, E, SE, S, SW, W, NW)
        movements = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        move_row, move_col = movements[action]

        # Calculate new position
        new_row = row + move_row
        new_col = col + move_col

        return new_row, new_col

    def get_position(self, state: int) -> Tuple[int, int]:
      """Convert 1D state back to 2D position (row, col)"""
      row = state // self.elevation_data.shape[1]
      col = state % self.elevation_data.shape[1]
      return (row, col)


    def get_valid_actions(self, state: int) -> List[int]:
        """Get list of valid actions for a given state"""
        position = self.get_position(state)
        valid_actions = []
        for action in range(8):
            next_pos = self._get_next_position(position, action)
            if 0 <= next_pos[0] < self.elevation_data.shape[0] and 0 <= next_pos[1] < self.elevation_data.shape[1]:
                valid_actions.append(action)
        return valid_actions if valid_actions else list(range(8))  # Return all actions if no valid actions


    def get_reward(self, current: Tuple[int, int], next: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate reward for moving from current to next position"""
        if next == goal:
            return 100  # High reward for reaching the goal

        # Negative reward based on terrain factors
        elevation_diff = abs(self.elevation_data[next] - self.elevation_data[current])
        obstacle_penalty = -50 if self.obstacle_data[next] > 0 else 0
        ruggedness_penalty = -self.ruggedness_data[next]

        # Encourage moving towards the goal
        current_to_goal = np.linalg.norm(np.array(goal) - np.array(current))
        next_to_goal = np.linalg.norm(np.array(goal) - np.array(next))
        progress_reward = current_to_goal - next_to_goal

        return progress_reward - elevation_diff - obstacle_penalty + ruggedness_penalty

    def visualize_episode(self, episode: int, path: List[Tuple[int, int]], reward: float, filename: str):
        """Visualize the path and reward for a specific episode"""
        save_dir = os.path.dirname(self.dem_file_path)
        save_path = os.path.join(save_dir, filename)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot the path on the terrain
        im = ax1.imshow(self.elevation_data.cpu(), cmap='terrain')
        if path:
            path_points = np.array(path)
            ax1.plot(path_points[:, 1], path_points[:, 0], 'r-', linewidth=2, label='Path')
        ax1.set_title(f'Episode {episode} Path')
        fig.colorbar(im, ax=ax1, label='Elevation (m)')
        ax1.legend()

        # Plot the rewards
        ax2.plot(range(len(self.episode_rewards)), self.episode_rewards, 'b-')
        ax2.set_title('Rewards per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.axhline(y=reward, color='r', linestyle='--', label=f'Current Episode Reward: {reward:.2f}')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Episode {episode} visualization saved to {save_path}")

    def train(self, start: Tuple[int, int], goal: Tuple[int, int], episodes: int = 1000, max_steps: int = 1000,
              learning_rate: float = 0.1, discount_factor: float = 0.99, epsilon: float = 0.1):
        """Train the Q-learning agent"""
        for episode in range(episodes):
            state = self.get_state(start)
            total_reward = 0
            path = [start]

            for step in range(max_steps):
                valid_actions = self.get_valid_actions(state)

                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    q_values = self.q_table[state][valid_actions]
                    action = valid_actions[torch.argmax(q_values).item()]

                next_position = self._get_next_position(self.get_position(state), action)
                next_state = self.get_state(next_position)
                reward = self.get_reward(self.get_position(state), next_position, goal)

                # Q-value update
                next_valid_actions = self.get_valid_actions(next_state)
                best_next_q_value = torch.max(self.q_table[next_state][next_valid_actions])
                td_target = reward + discount_factor * best_next_q_value
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += learning_rate * td_error

                total_reward += reward
                state = next_state
                path.append(next_position)

                if next_position == goal:
                    break

            self.episode_rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
                self.visualize_episode(episode, path, total_reward, f"episode_{episode}_visualization.png")


    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path using trained Q-table"""
        path = [start]
        current = start

        while current != goal:
            state = self.get_state(current)
            action = torch.argmax(self.q_table[state]).item()
            next_position = self._get_next_position(current, action)
            path.append(next_position)
            current = next_position

            if len(path) > 1000:  # Prevent infinite loop
                print("Path too long, stopping search.")
                break

        return path

    def save_path(self, path: List[Tuple[int, int]], filename: str):
        """Save the path to a CSV file"""
        if path:
            save_dir = os.path.dirname(self.dem_file_path)
            save_path = os.path.join(save_dir, filename)
            np.savetxt(save_path, path, delimiter=',', fmt='%d')
            print(f"Path saved to {save_path}")
        else:
            print(f"No path to save for {filename}")

    def save_path_image(self, path: List[Tuple[int, int]], filename: str):
        """Save the path as an image"""
        if path:
            save_dir = os.path.dirname(self.dem_file_path)
            save_path = os.path.join(save_dir, filename)

            plt.figure(figsize=(12, 10))
            plt.imshow(self.elevation_data.cpu(), cmap='terrain')

            if path:
                path_points = np.array(path)
                plt.plot(path_points[:, 1], path_points[:, 0], 'r-', linewidth=2, label='Path')

            plt.colorbar(label='Elevation (m)')
            plt.title(f'RL Path on Terrain')
            plt.legend()

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Path image saved to {save_path}")
        else:
            print(f"No path to save as image for {filename}")

def main():
    dem_path = '/content/drive/MyDrive/path-ril/path.tif'

    # Create pathfinder with pixel reduction
    pathfinder = TerrainRLPathfinder(dem_path, reduction_factor=0.5)

    # Preprocess and calculate all factors
    pathfinder.preprocess_dem()

    # Initialize Q-table
    pathfinder.initialize_q_table()

    # Define start and goal
    start = (50, 50)  # Adjust based on reduced image size
    goal = (200, 200)  # Adjust based on reduced image size

    print("Training RL agent...")
    pathfinder.train(start, goal, episodes=5000, max_steps=1000)

    print("Finding path...")
    rl_path = pathfinder.find_path(start, goal)

    # Save path as CSV
    pathfinder.save_path(rl_path, "rl_path.csv")

    # Save path as image
    pathfinder.save_path_image(rl_path, "rl_path.png")

    print("Path finding complete.")

if __name__ == "__main__":
    main()
