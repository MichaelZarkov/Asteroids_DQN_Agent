import torch

import asteroids_env as ast
import fun_env as fun
from train import DQN

def demo(which_env):
    """Watch the DQN agent play the environment."""
    if which_env == 'fun': env = fun.Space(training_mode=False, render_mode='human')
    else:                  env = ast.Space(training_mode=False, render_mode='human')

    # Load model from file.
    DQN_model = DQN(env.observation_space.shape[0], env.action_space.n)
    DQN_model.load_state_dict(torch.load('../models/model_final/not_final_but_good_enough.pt'))  # Load model from file.
    DQN_model.eval()  # Set model to evaluation mode.

    # Game loop.
    score = 0
    observation, _ = env.reset()
    while True:
        obs_tensor = torch.from_numpy(observation)  # Convert numpy array to tensor.
        action = torch.argmax(DQN_model(obs_tensor))  # Let the agent choose an action.
        observation, reward, terminated, _, _ = env.step(action)  # Take the action in the env.
        score += reward
        if terminated: break
    print("Final score: ", score)

if __name__ == '__main__':
    demo(which_env="fun")