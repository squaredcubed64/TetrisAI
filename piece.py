from typing import List, Set, Tuple
from piece_type import PieceType
from rotation import Rotation

class Piece:
    def __init__(self, type: PieceType, top_left: Tuple[int, int], rotation: Rotation = Rotation.ZERO):
        self.type = type
        self.top_left = top_left # top_left
        self.rotation = rotation
    
    def move(self, offset: Tuple[int, int]) -> None:
        self.top_left = (self.top_left[0] + offset[0], self.top_left[1] + offset[1])
    
    def rotate_counterclockwise(self) -> None:
        self.rotation = self.rotation.counterclockwise()
    
    def rotate_clockwise(self) -> None:
        self.rotation = self.rotation.clockwise()
    
    @staticmethod
    def rotate_body_clockwise(body: Set[Tuple[int, int]], bounding_box_size: int) -> Set[Tuple[int, int]]:
        return {(bounding_box_size - 1 - y, x) for x, y in body}

    def get_cells(self) -> Set[Tuple[int, int]]:
        cells_relative_to_top_left = set()
        if self.rotation == Rotation.ZERO:
            cells_relative_to_top_left = self.type.up_body
        elif self.rotation == Rotation.RIGHT:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size)
        elif self.rotation == Rotation.TWO:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size), self.type.bounding_box_size)
        elif self.rotation == Rotation.LEFT:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size), self.type.bounding_box_size), self.type.bounding_box_size)
        return {(x + self.top_left[0], y + self.top_left[1]) for x, y in cells_relative_to_top_left}
    
    def is_out_of_bounds(self, board_width: int, board_height: int) -> bool:
        for x, y in self.get_cells():
            if x < 0 or x >= board_width or y < 0 or y >= board_height:
                return True
        return False

    # indexError if piece is out of bounds
    def is_colliding_with_stack(self, stack: List[List[int]]) -> bool:
        for x, y in self.get_cells():
            if stack[y][x]:
                return True
        return False
    
    def is_colliding_or_out_of_bounds(self, stack: List[List[int]]) -> bool:
        return self.is_out_of_bounds(len(stack[0]), len(stack)) or self.is_colliding_with_stack(stack)
    
    def place_on_stack(self, stack: List[List[int]]) -> None:
        for x, y in self.get_cells():
            stack[y][x] = 1
    
    def has_same_cells(self, other: 'Piece') -> bool:
        return self.get_cells() == other.get_cells()

    def __hash__ (self) -> int:
        return hash((self.type, self.top_left, self.rotation))

    def __eq__(self, other: 'Piece') -> bool:
        return self.type == other.type and self.top_left == other.top_left and self.rotation == other.rotation
