from enum import Enum
from typing import Dict, Set, Tuple
from rotation import Rotation

JLSTZ_WALL_KICKS = {
    Rotation.ZERO: {
        Rotation.RIGHT: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        Rotation.LEFT: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)]
    },
    Rotation.RIGHT: {
        Rotation.ZERO: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        Rotation.TWO: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
    },
    Rotation.TWO: {
        Rotation.RIGHT: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        Rotation.LEFT: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)]
    },
    Rotation.LEFT: {
        Rotation.ZERO: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        Rotation.TWO: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)]
    }
}

I_WALL_KICKS = {
    Rotation.ZERO: {
        Rotation.RIGHT: [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        Rotation.LEFT: [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)]
    },
    Rotation.RIGHT: {
        Rotation.ZERO: [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        Rotation.TWO: [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)]
    },
    Rotation.TWO: {
        Rotation.RIGHT: [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        Rotation.LEFT: [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)]
    },
    Rotation.LEFT: {
        Rotation.ZERO: [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        Rotation.TWO: [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)]
    }
}

O_WALL_KICKS = {
    Rotation.ZERO: {
        Rotation.RIGHT: [(0, 0)],
        Rotation.LEFT: [(0, 0)]
    },
    Rotation.RIGHT: {
        Rotation.ZERO: [(0, 0)],
        Rotation.TWO: [(0, 0)]
    },
    Rotation.TWO: {
        Rotation.RIGHT: [(0, 0)],
        Rotation.LEFT: [(0, 0)]
    },
    Rotation.LEFT: {
        Rotation.ZERO: [(0, 0)],
        Rotation.TWO: [(0, 0)]
    }
}

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BLUE = (0, 255, 255)
DARK_BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 128, 0)
RED = (128, 0, 0)
MAGENTA = (165, 0, 165)

class PieceType(Enum):
    EMPTY = (BLACK, None, None, None)
    I = (LIGHT_BLUE, 4, {(0, 1), (1, 1), (2, 1), (3, 1)}, I_WALL_KICKS)
    J = (DARK_BLUE, 3, {(0, 0), (0, 1), (1, 1), (2, 1)}, JLSTZ_WALL_KICKS)
    L = (ORANGE, 3, {(2, 0), (0, 1), (1, 1), (2, 1)}, JLSTZ_WALL_KICKS)
    O = (YELLOW, 2, {(0, 0), (1, 0), (0, 1), (1, 1)}, O_WALL_KICKS)
    S = (GREEN, 3, {(1, 0), (2, 0), (0, 1), (1, 1)}, JLSTZ_WALL_KICKS)
    Z = (RED, 3, {(0, 0), (1, 0), (1, 1), (2, 1)}, JLSTZ_WALL_KICKS)
    T = (MAGENTA, 3, {(1, 0), (0, 1), (1, 1), (2, 1)}, JLSTZ_WALL_KICKS)

    def __init__(self, color: Tuple[int, int, int], bounding_box_size: int, up_body: Set[Tuple[int, int]], wall_kicks: Dict[Rotation, Dict[Rotation, Set[Tuple[int, int]]]]):
        self.color = color
        self.bounding_box_size = bounding_box_size
        self.up_body = up_body
        self.wall_kicks = wall_kicks

    def __str__(self) -> str:
        if self == PieceType.EMPTY:
            return "."
        else:
            return self.name
