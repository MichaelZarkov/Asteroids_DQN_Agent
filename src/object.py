import math
import random as rnd

class Object:
    # Must be set prior to creating object instance.
    FW = None  # Field width (int).
    FH = None  # Field height (int).
    @staticmethod
    def set_width_and_height(width, height):
        Object.FW = width
        Object.FH = height

    def __init__(self, hb_size, pos=None, edge=None, speed=None, speed_range=None, dir=None):
        """
        :param hb_size: float, hit box size. Objects are circles and this is the radius of the hit bix.
        :param pos: (float, float), position on the field.
        :param edge: bool, where to spawn if 'pos' is not specified. False - spawn anywhere, True - spawn on edge of field.
        :param speed: float, in units per second. If speed == FW it will take 1 second to cross the field.
        :param speed_range: (float, float), is 'speed' is not specified, give random speed in interval.
        :param dir: (float, float), direction of travel. If not specified give random direction.
        """
        self.hb_size = hb_size
        self.pos = pos if pos is not None else Object._rand_pos(edge)
        self.speed = speed if speed is not None else rnd.uniform(speed_range[0], speed_range[1])
        self.dir = Object._normalize(dir) if dir is not None else Object._rand_dir()

    @staticmethod
    def _rand_pos(edge):
        """Returns a random position in the specified rectangle. If 'edge' is True, the position is at
        the edge of the field. Distribution is uniform."""
        if edge:
            side = rnd.randint(1, 4)  # Which side.
            if side == 1:
                x, y =  0.0, rnd.randint(0, Object.FH - 1)
            elif side == 2:
                x, y = Object.FW - 1, rnd.randint(0, Object.FH - 1)
            elif side == 3:
                x, y = rnd.randint(0, Object.FW - 1), 0.0
            else:
                x, y = rnd.randint(0, Object.FW - 1), Object.FH - 1
        else:
            x, y = rnd.randint(0, Object.FW - 1), rnd.randint(0, Object.FH-1)
        return float(x), float(y)

    @staticmethod
    def _rand_dir():
        """Returns a random direction - random normalized vector."""
        x = rnd.uniform(-1.0, 1.0)
        y = math.sqrt(1.0 - x * x) * (-1.0 if rnd.randint(0, 1) else 1.0)
        return x, y

    @staticmethod
    def _normalize(v):
        """Normalizes a 2D vector."""
        length = math.sqrt(v[0]*v[0] + v[1]*v[1])
        return v[0]/length, v[1]/length

    def update_pos(self, dt):
        """Updates the position of the object. The speed is measured in units per second. So 'dt' must be in seconds.
        For example if 'dt == 0.01' that means we're running with 100fps and the distance traveled will be 1/100 of
        'self.speed'. Objects wrap around to the other side of the field if they cross the border."""
        length = dt * self.speed
        x_new, y_new = (self.pos[0] + self.dir[0]*length, self.pos[1] + self.dir[1]*length)
        if x_new < 0.0:
            x_new = Object.FW
        elif x_new > Object.FW:
            x_new = 0.0

        if y_new < 0.0:
            y_new = Object.FH
        elif y_new > Object.FH:
            y_new = 0.0

        self.pos = (x_new, y_new)