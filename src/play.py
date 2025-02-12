from gymnasium.utils.play import play
import torch

from asteroids_env import Space
from train import DQN

def play_asteroids():
    """Play the Asteroids environment."""
    env = Space(training_mode=True, render_mode='rgb_array')
    play(env, fps=Space.METADATA['render_fps'], keys_to_action=Space.KEY_ACTION_MAP)

def DQN_play():
    """Watch the DQN play the environment."""
    env = Space(training_mode=False, render_mode='human')
    # Load model from file.
    DQN_model = DQN(env.observation_space.shape[0], env.action_space.n)
    DQN_model.load_state_dict(torch.load('../models/910_policy_net.pt'))  # Load model from file.
    DQN_model.eval()  # Set model to evaluation mode.

    # Game loop.s
    score = 0
    observation, _ = env.reset()
    while True:
        obs_tensor = torch.from_numpy(observation)  # Convert numpy array to tensor.
        action = torch.argmax(DQN_model(obs_tensor))  # Let the agent choose an action.
        observation, reward, terminated, _, _ = env.step(action)  # Take the action in the env.
        score += reward
        if terminated: break
    print("Final score: ", score)

play_asteroids()
#DQN_play()