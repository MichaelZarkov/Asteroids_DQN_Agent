import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import asteroids_env

device = "cpu"  # Set device for PyTorch.

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# The neural network for the DQN agent.
class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 96)
        self.layer3 = nn.Linear(96, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPSILON is the discount factor.
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.985
EPSILON = 0.1
TAU = 0.005
LR = 1e-4
REPLAY_MEMORY_CAPACITY = 2500000
EPISODES = 2000

env = asteroids_env.Space(training_mode=True, render_mode=None)
# Get number of actions from gym action space.
n_actions = env.action_space.n
# Get the number of state observations.
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)  # This is basically the amount of experiences we store.

def select_action(state, episode_number):
    """Select action according to epsilon greedy policy."""
    sample = random.random()  # Get uniform rand number 0.0 <= x < 1.0
    if sample > EPSILON:
        # Return the network prediction.
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # Else, explore by returning random action.
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_and_save(episode_number, ys, title, x_label, y_label):
    plt.close()  # Clear the plot.
    plt.plot(ys, color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if episode_number >= 30:
        hundred_average = [0 for i in range(29)]
        for i in range(0, len(ys) - 29):
            hundred_average += [sum(ys[i:i + 30]) / 30]
        plt.plot(hundred_average, color='orange')

    plt.savefig(f'../models/tmp/plots/{title}_{episode_number}.png')

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended).
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values.
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss.
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model.
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping.
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == '__main__':
    episode_durations = []  # In seconds.
    episode_rewards = []

    for i_episode in range(EPISODES):
        # Initialize the environment and get its state.
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0  # Count for the total reward in this episode.
        for t in count():
            action = select_action(state, i_episode)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print(f"Episode: {i_episode};",
                      f"Reward: {total_reward};",
                      f"Duration: {t // asteroids_env.Space.METADATA['render_fps']} sec.")
                episode_durations += [t / asteroids_env.Space.METADATA['render_fps']]
                episode_rewards += [total_reward]
                break

        # Save every episode.
        torch.save(target_net.state_dict(), f'../models/tmp/nets/{i_episode}_target_net.pt')
        torch.save(policy_net.state_dict(), f'../models/tmp/nets/{i_episode}_policy_net.pt')
        # Plot every episode.
        plot_and_save(i_episode, episode_durations, title='Durations', x_label='Episode', y_label='Duration in seconds')
        plot_and_save(i_episode, episode_rewards, title='Rewards', x_label='Episode', y_label='Reward')

    print('Complete')