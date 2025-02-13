from gymnasium.utils.play import play

import asteroids_env as ast

def play_env():
    """Play the Asteroids environment."""
    env = ast.Space(training_mode=False, render_mode='rgb_array')
    play(env, fps=env.METADATA['render_fps'], keys_to_action=env.KEY_ACTION_MAP)

if __name__ == '__main__':
    play_env()
