from object import Object

class Projectile(Object):
    # Defaults.
    _SPEED = 500.0
    _HB_SIZE = 2.0
    _LIFETIME = 350.0  # Measure of distance.

    def __init__(self, pos, dir):
        super().__init__(hb_size=Projectile._HB_SIZE, pos=pos, speed=Projectile._SPEED, dir=dir)

        # Represents the lifetime of the projectile. It is a measure of distance.
        # For example if 'self.max_lifetime' is 1000.0 then the projectile should despawn when
        # 'self.lifetime' reaches 0.0.
        self.lifetime = Projectile._LIFETIME

    def is_alive(self):
        return self.lifetime > 0.0

    def update_pos(self, dt):
        super().update_pos(dt)
        self.lifetime -= dt * self.speed