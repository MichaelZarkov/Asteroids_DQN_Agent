from gymnasium.utils.play import play

from asteroids_env import Space

if __name__ == '__main__':
    """Play the Asteroids environment."""
    env = Space(training_mode=True, render_mode='rgb_array')  # Play in training mode to see the death replay feature.
    play(env, fps=Space.METADATA['render_fps'], keys_to_action=Space.KEY_ACTION_MAP)