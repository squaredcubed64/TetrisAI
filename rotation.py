# source for rotations: https://tetris.wiki/Super_Rotation_System
from enum import Enum

class Rotation(Enum):
    ZERO = 0
    RIGHT = 1
    TWO = 2
    LEFT = 3

    def clockwise(self) -> 'Rotation':
        return Rotation((self.value + 1) % 4)

    def counterclockwise(self) -> 'Rotation':
        return Rotation((self.value - 1) % 4)
