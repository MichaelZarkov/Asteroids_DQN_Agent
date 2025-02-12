## Model specs

### Neural Network

```python
# The neural network for the DQN agent.
class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

### Learning Parameters

```python
# BATCH_SIZE is the number of transitions sampled from the replay buffer.
# GAMMA is the discount factor as mentioned in the previous section.
# EPSILON is the discount factor.
# TAU is the update rate of the target network.
# LR is the learning rate of the ``AdamW`` optimizer.
BATCH_SIZE = 128
GAMMA = 0.985
EPSILON = 0.1
TAU = 0.005
LR = 1e-4

REPLAY_MEMORY_CAPACITY = 2000000  # 2 million
EPISODES = 1000
```

### Environment Parameters

```python
RENDER_FPS = 30

BORDER_SIZE = 52
WW = 700  # Window width.
WH = 700  # Window height.
FW = WW + 2 * BORDER_SIZE  # Game field width.
FH = WH + 2 * BORDER_SIZE  # Game field height.

TRAINING_LIVES = 1  # Number of allowed player-asteroid collisions before termination.
RELIVE_STATES = 45   # Save this many states in the past.

MAX_ASTEROIDS = 1  # Don't spawn if more than this.
SPAWN_TIME = 0.1   # Interval between asteroid spawns measured in second.

MAX_OBS_AST = 1  # Max observable asteroids. The observation includes at most this many asteroids.
MAX_OBS_PRJ = 2  # Max observable projectiles.
```