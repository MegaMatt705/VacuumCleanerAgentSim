import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Custom colormap from white to red
white_to_red = LinearSegmentedColormap.from_list("WhiteToRed", ["white", "red"])

class Environment:
    def __init__(self, width, height, dirt_percentage):
        self.width = width
        self.height = height
        self.floor = np.zeros((height, width))
        self.initialize_dirt(dirt_percentage)

    def initialize_dirt(self, dirt_percentage):
        num_tiles = self.width * self.height
        num_dirty = int(num_tiles * dirt_percentage / 100)
        for _ in range(num_dirty):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.floor[y, x] += 1  # Increase dirt level

    def clean_tile(self, x, y):
        if self.floor[y, x] > 0:
            self.floor[y, x] -= 1  # Clean a level of dirt

    def is_dirty(self, x, y):
        return self.floor[y, x] > 0

    def get_performance_measure(self):
        return np.sum(self.floor)  # Total amount of dirt left as the performance measure

class Agent:
    def __init__(self, env):
        self.env = env
        self.x = env.width // 2
        self.y = env.height // 2
        self.actions_taken = 0

    def move(self, direction):
        if direction == 'up' and self.y > 0:
            self.y -= 1
        elif direction == 'down' and self.y < self.env.height - 1:
            self.y += 1
        elif direction == 'left' and self.x > 0:
            self.x -= 1
        elif direction == 'right' and self.x < self.env.width - 1:
            self.x += 1
        self.actions_taken += 1

    def suck(self):
        self.env.clean_tile(self.x, self.y)
        self.actions_taken += 1

    def action(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class RandomAgent(Agent):
    def action(self):
        action = random.choice(['up', 'down', 'left', 'right', 'suck'])
        if action == 'suck':
            self.suck()
        else:
            self.move(action)

class ReflexAgent(Agent):
    def action(self):
        if self.env.is_dirty(self.x, self.y):
            self.suck()
        else:
            self.move(random.choice(['up', 'down', 'left', 'right']))

class ModelBasedReflexAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.known_dirt = set()
        self.visit_counts = np.zeros((env.height, env.width))  # Track visits to each grid

    def update_known_dirt(self):
        # Mark current location as visited and update dirt knowledge
        self.visit_counts[self.y, self.x] += 1
        if self.env.is_dirty(self.x, self.y):
            self.known_dirt.add((self.x, self.y))
        else:
            self.known_dirt.discard((self.x, self.y))  # Remove from known dirt if cleaned

    def action(self):
        self.update_known_dirt()
        if self.env.is_dirty(self.x, self.y):
            self.suck()
        elif self.known_dirt:
            # Move to the closest known dirty location
            target_x, target_y = self.choose_least_visited(self.known_dirt)
            self.move_to(target_x, target_y)
        else:
            # If no known dirt, move to the least visited cell
            least_visited = np.unravel_index(np.argmin(self.visit_counts), self.visit_counts.shape)
            self.move_to(*least_visited)

    def move_to(self, target_x, target_y):
        # Simplified logic to move towards a target; could be enhanced for more direct paths
        if target_x > self.x:
            self.move('right')
        elif target_x < self.x:
            self.move('left')
        elif target_y > self.y:
            self.move('down')
        elif target_y < self.y:
            self.move('up')

    def choose_least_visited(self, locations):
        # Choose the known dirty location that has been visited the least
        return min(locations, key=lambda loc: self.visit_counts[loc[1], loc[0]])

def simulate_agent(agent_class, width, height, dirt_percentage, steps):
    env = Environment(width, height, dirt_percentage)
    agent = agent_class(env)
    initial_dirt = env.get_performance_measure()
    before_state = np.copy(env.floor)
    for _ in range(steps):
        agent.action()
    after_state = np.copy(env.floor)
    cleaned_dirt = initial_dirt - env.get_performance_measure()
    return cleaned_dirt, agent.actions_taken, before_state, after_state

def visualize_environment(before, after, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Define a more contrasting colormap directly in the visualization function
    cmap = LinearSegmentedColormap.from_list("WhiteToRed", ["white", "red"])
    
    max_dirt_level = max(np.max(before), np.max(after))  # Find the maximum dirt level for normalization
    if max_dirt_level == 0:  # Prevent division by zero in case of no dirt
        max_dirt_level = 1
    
    # Normalize the colormap to the range of dirt levels present for clearer differentiation
    norm = plt.Normalize(vmin=0, vmax=max_dirt_level)
    
    axs[0].imshow(before, cmap=cmap, interpolation='nearest', norm=norm)
    axs[0].set_title(f'{title} Before')
    axs[1].imshow(after, cmap=cmap, interpolation='nearest', norm=norm)
    axs[1].set_title(f'{title} After')
    
    # Adding a colorbar to indicate dirt levels
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    
    for ax in axs:
        ax.axis('off')
    plt.show()

def run_simulations_and_report(grid_sizes, dirt_percentage, steps=1000, runs=1):
    agents = [RandomAgent, ReflexAgent, ModelBasedReflexAgent]
    results = []

    for size in grid_sizes:
        for agent_class in agents:
            performance = []
            for _ in range(runs):
                cleaned_dirt, actions_taken, before_state, after_state = simulate_agent(agent_class, size, size, dirt_percentage, steps)
                performance.append(cleaned_dirt)
                visualize_environment(before_state, after_state, f"{agent_class.__name__} on {size}x{size}")
            avg_performance = sum(performance) / runs
            results.append({'Grid Size': f'{size}x{size}', 'Agent Type': agent_class.__name__, 'Average Dirt Cleaned': avg_performance})
    
    df = pd.DataFrame(results)
    print(df.pivot("Grid Size", "Agent Type", "Average Dirt Cleaned"))

# Example simulation runs
grid_sizes = [5, 50, 100]
run_simulations_and_report(grid_sizes, dirt_percentage=20)
