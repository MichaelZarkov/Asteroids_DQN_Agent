from object import Object

class Asteroid(Object):
    # Corresponds to the size of the asteroid.
    TYPES = ['S','M','L']

    S = {'speed_range': (120.0,150.0), 'hb_size': 15.0}
    M = {'speed_range': (50.0,120.0), 'hb_size': 30.0}
    L = {'speed_range': (40.0,80.0), 'hb_size': 50.0}

    def __init__(self, type, pos=None):
        assert type in Asteroid.TYPES, "Invalid asteroid type!"
        self.type = type

        if self.type == 'S':
            sr = Asteroid.S['speed_range']
            hb = Asteroid.S['hb_size']
        elif self.type == 'M':
            sr = Asteroid.M['speed_range']
            hb = Asteroid.M['hb_size']
        else:
            sr = Asteroid.L['speed_range']
            hb = Asteroid.L['hb_size']

        super().__init__(hb_size=hb, pos=pos, edge=True, speed_range=sr)

    def _split_M(self):
        """Returns smaller asteroids as if it was split."""
        return [Asteroid(type='S', pos=self.pos), Asteroid(type='S', pos=self.pos), Asteroid(type='S', pos=self.pos)]

    def _split_L(self):
        """Returns smaller asteroids as if it was split."""
        return [Asteroid(type='M', pos=self.pos), Asteroid(type='M', pos=self.pos), Asteroid(type='M', pos=self.pos)]

    def split(self):
        if self.type == 'S': return []  # The smallest asteroid cannot be split.
        elif self.type == 'M': return self._split_M()
        else: return self._split_L()