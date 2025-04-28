import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

# Check if MPS (Metal Performance Shaders) is available for M3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class GoBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.history = []
        self.ko_point = None
        self.passes = 0  # Track consecutive passes
        self.max_moves = size * size * 2  # Maximum number of moves before forcing end
        
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.history = []
        self.ko_point = None
        self.passes = 0
        return self.get_state()
    
    def get_state(self):
        # Create a 3-channel state representation
        state = np.zeros((3, self.size, self.size))
        state[0] = (self.board == 1).astype(float)  # Black stones
        state[1] = (self.board == -1).astype(float)  # White stones
        state[2] = self.current_player  # Current player
        return state
    
    def is_valid_move(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x, y] != 0:
            return False
        if (x, y) == self.ko_point:
            return False
        return True
    
    def get_liberties(self, x, y):
        color = self.board[x, y]
        if color == 0:
            return set()
        
        group = set()
        liberties = set()
        to_check = [(x, y)]
        
        while to_check:
            current = to_check.pop()
            if current in group:
                continue
            group.add(current)
            cx, cy = current
            
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == 0:
                        liberties.add((nx, ny))
                    elif self.board[nx, ny] == color:
                        to_check.append((nx, ny))
        
        return liberties
    
    def remove_group(self, x, y):
        color = self.board[x, y]
        group = set()
        to_remove = set()
        to_check = [(x, y)]
        
        while to_check:
            current = to_check.pop()
            if current in group:
                continue
            group.add(current)
            cx, cy = current
            
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] == color:
                        to_check.append((nx, ny))
        
        for x, y in group:
            self.board[x, y] = 0
            to_remove.add((x, y))
        
        return to_remove
    
    def make_move(self, x, y):
        # Handle pass move (x=-1, y=-1)
        if x == -1 and y == -1:
            self.passes += 1
            self.current_player *= -1
            return True
            
        if not self.is_valid_move(x, y):
            return False
        
        self.passes = 0  # Reset passes counter on valid move
        self.board[x, y] = self.current_player
        captured = set()
        
        # Check for captures
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.board[nx, ny] == -self.current_player:
                    liberties = self.get_liberties(nx, ny)
                    if not liberties:
                        captured.update(self.remove_group(nx, ny))
        
        # Check if move is suicidal
        if not self.get_liberties(x, y) and not captured:
            self.board[x, y] = 0
            return False
        
        # Update ko point
        if len(captured) == 1:
            self.ko_point = list(captured)[0]
        else:
            self.ko_point = None
        
        self.history.append((x, y))
        self.current_player *= -1
        return True
    
    def get_reward(self):
        # Simple scoring: count territory
        black_territory = np.sum(self.board == 1)
        white_territory = np.sum(self.board == -1)
        return black_territory - white_territory
    
    def is_game_over(self):
        # Game ends if:
        # 1. Both players pass consecutively
        # 2. Maximum number of moves reached
        return self.passes >= 2 or len(self.history) >= self.max_moves

class GoNet(nn.Module):
    def __init__(self, board_size=9):
        super(GoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Policy head
        policy = torch.relu(self.policy_conv(x))
        policy = self.policy_fc(policy.view(-1, 32 * 9 * 9))
        policy = torch.softmax(policy, dim=1)
        
        # Value head
        value = torch.relu(self.value_conv(x))
        value = torch.relu(self.value_fc1(value.view(-1, 32 * 9 * 9)))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class GoAgent:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.network = GoNet(board_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.debug = False  # Enable/disable debugging
        self.train_interval = 4  # Train every N steps
        self.steps = 0  # Track total steps
    
    def save(self, path):
        """Save the agent's network and optimizer state"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load the agent's network and optimizer state"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.eval()  # Set to evaluation mode
    
    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            # Include pass move (-1, -1) in random actions
            if random.random() < 0.1:  # 10% chance to pass
                return -1
            return random.randint(0, self.board_size * self.board_size - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, _ = self.network(state)
        return torch.argmax(policy).item()
    
    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0.0
        
        # Only train every N steps
        self.steps += 1
        if self.steps % self.train_interval != 0:
            return 0.0
        
        # Sample batch and convert to numpy arrays first
        batch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Get current Q values
        current_policy, current_value = self.network(states)
        current_q = current_policy.gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_policy, next_value = self.network(next_states)
            next_q = next_value
        
        # Compute target values
        target_values = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        policy_loss = -torch.mean(current_q)
        value_loss = torch.mean((target_values - current_value) ** 2)
        total_loss = policy_loss + value_loss
        
        if self.debug:
            print(f"\nTraining Debug Info:")
            print(f"Average Reward: {rewards.mean().item():.2f}")
            print(f"Average Current Value: {current_value.mean().item():.2f}")
            print(f"Average Target Value: {target_values.mean().item():.2f}")
            print(f"Policy Loss: {policy_loss.item():.4f}")
            print(f"Value Loss: {value_loss.item():.4f}")
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

def train_agents(episodes=1000, batch_size=64, debug=False, save_path=None):
    board = GoBoard()
    agent1 = GoAgent()
    agent2 = GoAgent()
    agent1.debug = debug
    agent2.debug = debug
    
    # Initialize experience buffers
    min_experiences = batch_size * 2
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = board.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        agent1_rewards = []  # Track rewards for agent 1
        agent2_rewards = []  # Track rewards for agent 2
        
        while not done:
            # Agent 1's turn (Black)
            action = agent1.get_action(state, epsilon=max(0.01, 0.1 - episode/1000))
            if action == -1:  # Pass move
                x, y = -1, -1
            else:
                x, y = action // 9, action % 9
                
            if board.make_move(x, y):
                next_state = board.get_state()
                reward = board.get_reward()  # Direct reward for Black
                agent1.memory.append((state, action, reward, next_state, False))
                state = next_state
                total_reward += reward
                agent1_rewards.append(reward)
                episode_steps += 1
                
                if debug and episode_steps % 10 == 0:
                    print(f"\nStep {episode_steps} - Agent 1 (Black) turn:")
                    print(f"Board reward: {reward}")
                    print(f"Black stones: {np.sum(board.board == 1)}")
                    print(f"White stones: {np.sum(board.board == -1)}")
            
            # Agent 2's turn (White)
            action = agent2.get_action(state, epsilon=max(0.01, 0.1 - episode/1000))
            if action == -1:  # Pass move
                x, y = -1, -1
            else:
                x, y = action // 9, action % 9
                
            if board.make_move(x, y):
                next_state = board.get_state()
                board_reward = board.get_reward()
                reward = -board_reward  # Negated reward for White
                agent2.memory.append((state, action, reward, next_state, False))
                state = next_state
                total_reward -= board_reward
                agent2_rewards.append(reward)
                episode_steps += 1
                
                if debug and episode_steps % 10 == 0:
                    print(f"\nStep {episode_steps} - Agent 2 (White) turn:")
                    print(f"Board reward: {board_reward}")
                    print(f"Agent 2 reward: {reward}")
                    print(f"Black stones: {np.sum(board.board == 1)}")
                    print(f"White stones: {np.sum(board.board == -1)}")
            
            # Check if game is over
            done = board.is_game_over()
        
        # Train both agents if enough experience is collected
        loss1 = agent1.train() if len(agent1.memory) >= min_experiences else 0
        loss2 = agent2.train() if len(agent2.memory) >= min_experiences else 0
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"Steps: {episode_steps}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Agent 1 (Black) - Average Reward: {np.mean(agent1_rewards):.2f}")
            print(f"Agent 2 (White) - Average Reward: {np.mean(agent2_rewards):.2f}")
            print(f"Loss1: {loss1:.4f}, Loss2: {loss2:.4f}")
            if debug:
                print(f"Agent 1 Reward Distribution: Min={min(agent1_rewards):.2f}, Max={max(agent1_rewards):.2f}")
                print(f"Agent 2 Reward Distribution: Min={min(agent2_rewards):.2f}, Max={max(agent2_rewards):.2f}")
                print(f"Final Board State:")
                print(f"Black stones: {np.sum(board.board == 1)}")
                print(f"White stones: {np.sum(board.board == -1)}")
                print(f"Memory sizes - Agent1: {len(agent1.memory)}, Agent2: {len(agent2.memory)}")
            
            # Save agents periodically
            if save_path and episode % 100 == 0:
                agent1.save(f"{save_path}/agent1_episode_{episode}.pth")
                agent2.save(f"{save_path}/agent2_episode_{episode}.pth")
    
    return agent1, agent2

if __name__ == "__main__":
    import os
    
    # Create directory for saving models
    save_dir = "trained_agents"
    os.makedirs(save_dir, exist_ok=True)
    
    # Train agents
    agent1, agent2 = train_agents(episodes=1000, save_path=save_dir)
    
    # Save final models
    agent1.save(f"{save_dir}/agent1_final.pth")
    agent2.save(f"{save_dir}/agent2_final.pth")
    
    # Play a match using the visualizer
    from go_visualizer import play_match
    play_match(agent1, agent2) 