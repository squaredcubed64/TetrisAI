from enum import Enum, auto

class Action(Enum):
    DO_NOTHING = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    ROTATE_CLOCKWISE = auto()
    ROTATE_COUNTERCLOCKWISE = auto()
    DROP = auto()