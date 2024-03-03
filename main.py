import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        q_values = self.model(torch.FloatTensor(state))
        if next_state is None:
            max_next_q_value = 0
        else:
            next_q_values = self.model(torch.FloatTensor(next_state))
            max_next_q_value = torch.max(next_q_values).item()  # Convert tensor to Python scalar

        target = torch.tensor(reward, dtype=torch.float32) + self.gamma * max_next_q_value * (1 - done)

        loss = nn.MSELoss()(q_values[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Game:
    def __init__(self):
        self.state_size = 4  # Game state: [ball_x, ball_y, paddle_x, done]
        self.action_size = 3  # Game actions: [move_left, stay, move_right]
        self.reset()

    def reset(self):
        self.ball_x = np.random.rand()
        self.ball_y = 1.0
        self.paddle_x = np.random.rand()
        self.done = False
        return [self.ball_x, self.ball_y, self.paddle_x, self.done]

    def step(self, action):
        if action == 0:  # Move left
            self.paddle_x = max(0, self.paddle_x - 0.1)
        elif action == 1:  # Stay
            pass
        elif action == 2:  # Move right
            self.paddle_x = min(1, self.paddle_x + 0.1)

        self.ball_y -= 0.1
        if self.ball_x < self.paddle_x:
            self.ball_x += 0.05
        elif self.ball_x > self.paddle_x:
            self.ball_x -= 0.05

        done = self.ball_y <= 0
        if done:
            if abs(self.ball_x - self.paddle_x) < 0.1:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0

        next_state = None if done else [self.ball_x, self.ball_y, self.paddle_x, done]
        return next_state, reward, done

# Hyperparameters
EPISODES = 10
MAX_STEPS = 100
STATE_SIZE = 4
ACTION_SIZE = 3

# Initialize game environment and agent
game = Game()
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Main training loop
for episode in range(EPISODES):
    state = game.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done = game.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
