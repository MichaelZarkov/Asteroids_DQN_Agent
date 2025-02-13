import copy
import gymnasium as gym
import math
import numpy as np
import pygame

from asteroid import Asteroid
from object import Object
from player import Player

class State:
    def __init__(self, spawn_timer, asteroids, projectiles, player):
        self.spawn_timer = spawn_timer
        self.asteroids = asteroids
        self.projectiles = projectiles
        self.player = player

class Space(gym.Env):
    """
    This environment is for f**king around and testing the agent's limits.
    If you break the code irreversibly, just copy and paste the 'asteroids_env.py' and start over.
    """

    METADATA = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    DT = 1.0 / METADATA['render_fps']  # Time between frames.
    # The actual game field should be slightly bigger than the display window. This is entirely
    # because of aesthetic reasons when viewing the game. If we don't have this difference between
    # the sizes (or if it is too small) we will see objects teleport from one edge of the window
    # to the other when they go over the edge of the field (objects wrap around to the other side when
    # they cross the edge).
    BORDER_SIZE = 15
    WW = 774  # Window width.
    WH = 774  # Window height.
    FW = WW + 2 * BORDER_SIZE  # Game field width.
    FH = WH + 2 * BORDER_SIZE  # Game field height.

    TRAINING_LIVES = 50  # Number of allowed player-asteroid collisions before termination.
    RELIVE_STATES = 60  # Save this many states in the past.

    MAX_ASTEROIDS = 3  # Don't spawn if more than this.
    SPAWN_TIME = 1.0  # Interval between asteroid spawns measured in second.

    MAX_OBS_AST = 1  # Max observable asteroids. The observation includes at most this many asteroids.

    AST_COL = (255, 255, 255)
    PL_COL = (255, 100, 100)
    PJ_COL = (130, 255, 130)
    SCREEN_COL = (0, 0, 0)

    # For human playing the game.
    KEY_ACTION_MAP = {
        (pygame.K_UP,): 1,
        (pygame.K_a,): 2,
        (pygame.K_d,): 3,
        (pygame.K_UP, pygame.K_a): 1,  # Prioritize shooting over rotating, when pressing 2 buttons.
        (pygame.K_UP, pygame.K_d): 1,  # Prioritize shooting over rotating.

        # Both rotate clockwise and rotate counterclockwise are pressed so they cancel each other out.
        (pygame.K_a, pygame.K_d): 0,
        (pygame.K_UP, pygame.K_a, pygame.K_d): 1,
    }

    def __init__(self, training_mode=False, render_mode=None):
        Object.set_width_and_height(Space.FW, Space.FH)

        # Set the initial state.
        self.s = State(spawn_timer=0.0, asteroids=[], projectiles=[], player=Player())

        """"
        An unsuccessful observation space. 
            # Observation is:
            # - Facing direction, shot cooldown of the player and binary input if shot cooldown is 0.
            # - Position and direction of travel of asteroid.
            # - Position, direction of travel and lifetime of projectiles.
            player_params = 2 + 1 + 1
            asteroid_params = Space.MAX_OBS_AST * (2 + 2)
            projectile_params = Space.MAX_OBS_PRJ * (2 + 2 + 1)
            total = player_params + asteroid_params + projectile_params
        """
        # - facing, cooldown, is_cooldown_ready - player
        # - pos, dir_travel, speed - asteroids
        player_params = 2 + 1 + 1
        asteroid_params = Space.MAX_OBS_AST * (2 + 2 + 1)
        total = player_params + asteroid_params
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)

        # Actions are DO_NOTHING, SHOOT, ROTATE_COUNTERCLOCKWISE, ROTATE_CLOCKWISE
        self.action_space = gym.spaces.Discrete(4)

        """
        Idea for training:
        This environment is one hit kill - when the agent hits an asteroid, it immediately dies and the environment
        is reset. On death we give negative reward to punish the agent for dying :). The problem is this reward is
        very very sparce. This makes the learning very slow. So instead of terminating immediately when colliding
        with an asteroid, we give it 'training_lives' allowed collisions before we terminate. We also save 
        'RELIVE_STATES' number of past states and on collision, we set the state to the oldest available so the
        agent has to 'relive' the terminating situation multiple times - hopefully figuring out a solution to avoid it.
        """
        self.training_mode = training_mode
        self.training_lives = Space.TRAINING_LIVES
        self.past_states = [copy.deepcopy(self.s) for _ in range(Space.RELIVE_STATES)]

        assert render_mode is None or render_mode in self.METADATA["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    @staticmethod
    def _intersect(x1, y1, x2, y2, r1, r2):
        """Returns true if the two circles with centers (x1,y1), (x2,y2) and radii r1, r2, intersect."""
        dx = x2 - x1
        dy = y2 - y1
        d = r1 + r2
        return d * d > dx * dx + dy * dy

    @staticmethod
    def _distance(x1, y1, x2, y2):
        """Returns the distance between two points."""
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _transform(point):
        """Transforms game field coordinates to display coordinates"""
        return point[0] - Space.BORDER_SIZE, point[1] - Space.BORDER_SIZE

    def _closest_asteroid(self):
        """Returns the closest asteroid to the player measured from the hit box of the asteroid."""
        if len(self.s.asteroids) == 0: return None
        ast = self.s.asteroids[0]
        dst = Space._distance(self.s.player.pos[0], self.s.player.pos[1], ast.pos[0], ast.pos[1]) - ast.hb_size
        for a in self.s.asteroids[1:]:
            new_dst = Space._distance(self.s.player.pos[0], self.s.player.pos[1], a.pos[0], a.pos[1]) - a.hb_size
            if dst > new_dst:
                ast = a
                dst = new_dst
        return ast

    def _get_obs(self):
        total_params = 4 + Space.MAX_OBS_AST * 5  # See __init__ for meaning of params.
        obs = np.zeros(shape=(total_params,), dtype=np.float32)

        # Player info.
        obs[0] = self.s.player.facing[0] + 1.2  # make them above 0.0 . Don't know if it helps.
        obs[1] = self.s.player.facing[1] + 1.2
        obs[2] = self.s.player.shot_cooldown
        obs[3] = 0.0 if self.s.player.shot_cooldown > 0.0 else 1.0

        # Asteroids info.
        # We kind of normalize the position coordinates to make them more suitable for a neural net.
        asts = [self._closest_asteroid()]
        if asts[0] is None: return obs
        i = 4
        for a in asts:
            obs[i] = a.pos[0] / Space.FW + 0.05
            obs[i + 1] = a.pos[1] / Space.FH + 0.05
            obs[i + 2] = a.dir[0] + 1.2
            obs[i + 3] = a.dir[1] + 1.2
            obs[i + 4] = a.speed / 150.0  # normalize this
            i += 5

        return obs

    def _precise_shot(self, projectile):
        """Simulate the environment and see if the given projectile will hit an asteroid during its lifetime.
        Give positive reward if it hist and negative otherwise.
        """
        # Copy because we will change them.
        a_cp = copy.deepcopy(self.s.asteroids)
        p = copy.deepcopy(projectile)  # Maybe just 'copy.copy(projectile)' will be enough here?
        while p.is_alive():
            # Check if the projectile hit an asteroid and update positions.
            for a in a_cp:
                if Space._intersect(a.pos[0], a.pos[1], p.pos[0], p.pos[1], a.hb_size, p.hb_size):
                    return 1  # Asteroid was hit - reward.
                a.update_pos(dt=Space.DT)
            p.update_pos(dt=Space.DT)
        return 0  # No asteroids were hit - no reward.

    def _player_act(self, action):
        """Executes the given action."""
        self.s.player.update_cooldown(dt=Space.DT)

        if action == 1:
            self.s.projectiles += self.s.player.shoot()
        elif action == 2:
            self.s.player.rotate(clockwise=False, dt=Space.DT)
        elif action == 3:
            self.s.player.rotate(clockwise=True, dt=Space.DT)

    def _player_collide(self):
        """Returns True if the player currently collides with an asteroid. Returns one of the asteroids the player
        currently collides with or None if the player doesn't collide with asteroid."""
        for a in self.s.asteroids:
            if Space._intersect(self.s.player.pos[0], self.s.player.pos[1], a.pos[0], a.pos[1], self.s.player.hb_size,
                                a.hb_size):
                return True, a
        return False, None

    def _projectile_collide(self):
        """Destroys the asteroids hit by projectiles. Despawns projectiles that reached their lifetime.
        Returns the number of destroyed asteroids."""
        new_projectiles = []
        count_destroyed = 0
        for p in self.s.projectiles:
            is_destroyed = False
            for a in self.s.asteroids:
                if Space._intersect(a.pos[0], a.pos[1], p.pos[0], p.pos[1], a.hb_size, p.hb_size):
                    self.s.asteroids.remove(a)
                    self.s.asteroids = self.s.asteroids + a.split()
                    is_destroyed = True
                    count_destroyed += 1
                    break
            if not is_destroyed and p.is_alive():
                new_projectiles.append(p)
        self.s.projectiles = new_projectiles
        return count_destroyed

    def _spawn_asteroid(self):
        """Spawns an asteroid randomly at the edge of the filed."""
        if self.s.spawn_timer <= 0.0 and len(self.s.asteroids) < Space.MAX_ASTEROIDS:
            self.s.asteroids.append(Asteroid('S'))
            self.s.spawn_timer = Space.SPAWN_TIME
        else:
            self.s.spawn_timer -= Space.DT

    # Note: See if 'seed' and 'options' are mandatory for the API and remove them if they are not.
    def reset(self, seed=None, options=None):
        """Reset the whole environment."""
        self.training_lives = self.TRAINING_LIVES
        self.s = State(spawn_timer=0.0, asteroids=[], projectiles=[], player=Player())
        self.past_states = [copy.deepcopy(self.s) for _ in range(Space.RELIVE_STATES)]

        observation = self._get_obs()
        info = dict()  # Not used at the moment.

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._player_act(action)
        # Award the player for being accurate.
        destroyed_asteroid_reward = self._precise_shot(
            projectile=self.s.projectiles[-1]) if self.s.player.shot_cooldown == Player._COOLDOWN else 0.0
        for a in self.s.asteroids: a.update_pos(dt=Space.DT)
        for p in self.s.projectiles: p.update_pos(dt=Space.DT)

        self._spawn_asteroid()
        self._projectile_collide()

        # We can play around with the reward to see what's optimal for the agent.
        # We give staying alive reward every frame and reward for projectile which is on path to hit an asteroid.
        staying_alive_reward = 0.0  # Currently no staying alive reward.
        total_reward = destroyed_asteroid_reward + staying_alive_reward
        if self.training_mode:
            collided, _ = self._player_collide()
            terminated = False
            # If we collide with asteroid, reset to a previous state moments before the collision so that the
            # agent can relive it and hopefully learn from it.
            if collided:
                total_reward = 0
                self.training_lives -= 1
                terminated = self.training_lives == 0  # Terminate if no more lives left.

                self.s = copy.deepcopy(self.past_states[0])  # Return in the past to relive the problem.
                self.past_states = self.past_states + [self.past_states[0]]
                self.past_states = self.past_states[1:]
            else:
                self.past_states = self.past_states + [copy.deepcopy(self.s)]
                self.past_states = self.past_states[1:]
        else:
            # If 'training_mode' is not ON, kill the player immediately (terminate).
            terminated, _ = self._player_collide()

        observation = self._get_obs()
        info = None
        if self.render_mode == "human": self._render_frame()
        return observation, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_player(self, canvas):
        # Draw the hit box for debugging.
        # pygame.draw.circle(
        #    canvas, color=AsteroidsEnv.PL_COL, center=AsteroidsEnv.transform(self.s.player.pos), radius=self.s.player.hb_size, width=2
        # )
        pygame.draw.polygon(
            canvas,
            color=Space.PL_COL,
            points=list(map(Space._transform, self.s.player.get_drawing())),
            width=2
        )

    def _draw_objects(self, canvas):
        """Draws every object on given canvas."""
        for a in self.s.asteroids:
            pygame.draw.circle(canvas, color=Space.AST_COL, center=Space._transform(a.pos), radius=a.hb_size, width=3)
        for p in self.s.projectiles:
            pygame.draw.circle(canvas, color=Space.PJ_COL, center=Space._transform(p.pos), radius=p.hb_size, width=2)
        self._draw_player(canvas)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((Space.WW, Space.WH))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((Space.WW, Space.WH))  # Isn't that like reeeealy inefficient?
        canvas.fill(Space.SCREEN_COL)
        self._draw_objects(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.METADATA["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()