from gymnasium.utils.play import play

import fun_env as fun

def play_env():
    """Play the Asteroids environment."""
    env = fun.Space(training_mode=False, render_mode='rgb_array')
    play(env, fps=env.METADATA['render_fps'], keys_to_action=env.KEY_ACTION_MAP)

if __name__ == '__main__':
    play_env()
