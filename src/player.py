import math

from object import Object
from projectile import Projectile

class Player(Object):
    # Defaults:
    _COOLDOWN = 1.0  # Shot cooldown. Measured in seconds.
    _FACING = (0.0,-1.0)  # Facing direction when spawned.
    _SPEED = 200.0
    _HB_SIZE = 25.0
    _ROTATION_RATE = 0.6  # Fraction of a circle per second. If 1.0 then the player can make one full rotation in one second.

    def __init__(self):
        # Note: Maybe later I can add an option to set different initial player parameters.
        super().__init__(hb_size=Player._HB_SIZE, pos=(Object.FW / 2, Object.FH / 2), speed=Player._SPEED)

        # Facing direction. This has to be a normal vector (length == 1).
        self.facing = Player._FACING
        # Measured in nanoseconds. Can shoot immediately after spawning.
        self.shot_cooldown = 0.0
        self.score = 0

    @staticmethod
    def _rotate_normal(v, beta, clockwise):
        """Rotates a normal vector by given angle in radians."""
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)
        if clockwise:
            return Object._normalize((v[0] * cos_beta - v[1] * sin_beta, v[1] * cos_beta + v[0] * sin_beta))
        else:
            return Object._normalize((v[0] * cos_beta + v[1] * sin_beta, v[1] * cos_beta - v[0] * sin_beta))

    @staticmethod
    def _scale(v, scalar):
        """Scale a 2D vector."""
        return v[0] * scalar, v[1] * scalar

    @staticmethod
    def _add(v1, v2):
        """Adds two 2D vectors."""
        return v1[0] + v2[0], v1[1] + v2[1]

    def update_cooldown(self, dt):
        """You must call this every frame. This eliminates the need for time keeping in 'Player' class."""
        if self.shot_cooldown <= 0.0: shot_cooldown = 0.0
        else: self.shot_cooldown -= dt

    def shoot(self):
        """Shoot a projectile at the direction we are facing. Returns the projectile."""
        if self.shot_cooldown > 0.0:
            return []  # Still in cooldown.
        self.shot_cooldown = Player._COOLDOWN
        # Spawn the projectile on the edge of our hit box.
        pj_pos = (self.pos[0] + self.hb_size*self.facing[0], self.pos[1] + self.hb_size*self.facing[1])
        pj_dir = self.facing
        return [Projectile(pos=pj_pos, dir=pj_dir)]

    def rotate(self, clockwise, dt):
        angle = 2 * math.pi * (Player._ROTATION_RATE * dt)  # In radians.
        self.facing = Player._rotate_normal(v=self.facing, beta=angle, clockwise=clockwise)

    def get_drawing(self):
        """Object is just a circle, but that is not very pleasing to the eye. This function returns the coordinates
        of a polygon which can be used to draw the player. The polygon looks like a pointy arrow. Of course this
        drawing is not representative of the real hit box which is a circle."""
        scale = 1.4 * self.hb_size
        angle = 3.1 * math.pi / 4.0
        v1 = Player._scale(self.facing, scale)  # The 'nose' point of our spaceship.
        v2 = Player._scale(Player._rotate_normal(v=self.facing, beta=angle, clockwise=False), scale)
        v3 = Player._scale(Player._rotate_normal(v=self.facing, beta=angle, clockwise=True), scale)
        p1 = Player._add(self.pos, v1)
        p2 = Player._add(self.pos, v2)
        p3 = Player._add(self.pos, v3)
        return p1, p2, self.pos, p3