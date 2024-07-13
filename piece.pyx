from libc.stdint cimport int32
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cpython cimport bool
from piece_type import PieceType
from rotation import Rotation

cdef class Piece:
    cdef public PieceType type
    cdef public pair[int, int] top_left
    cdef public Rotation rotation

    def __init__(self, PieceType type, pair[int, int] top_left, Rotation rotation=Rotation.ZERO):
        self.type = type
        self.top_left = top_left
        self.rotation = rotation
    
    cpdef move(self, pair[int, int] offset):
        self.top_left = (self.top_left.first + offset.first, self.top_left.first + offset.first)
    
    cpdef rotate_counterclockwise(self):
        self.rotation = self.rotation.counterclockwise()
    
    cpdef rotate_clockwise(self):
        self.rotation = self.rotation.clockwise()
    
    @staticmethod
    cdef set[pair[int, int]] rotate_body_clockwise(set[pair[int, int]] body, int bounding_box_size):
        return {(bounding_box_size - 1 - y, x) for x, y in body}

    cpdef set[pair[int, int]] get_cells(self):
        cdef set[pair[int, int]] cells_relative_to_top_left
        if self.rotation == Rotation.ZERO:
            cells_relative_to_top_left = self.type.up_body
        elif self.rotation == Rotation.RIGHT:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size)
        elif self.rotation == Rotation.TWO:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size), self.type.bounding_box_size)
        elif self.rotation == Rotation.LEFT:
            cells_relative_to_top_left = Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(Piece.rotate_body_clockwise(self.type.up_body, self.type.bounding_box_size), self.type.bounding_box_size), self.type.bounding_box_size)
        return {(x + self.top_left.first, y + self.top_left.second) for x, y in cells_relative_to_top_left}
    
    def is_out_of_bounds(self, board_width: int, board_height: int) -> bool:
        for x, y in self.get_cells():
            if x < 0 or x >= board_width or y < 0 or y >= board_height:
                return True
        return False

    # indexError if piece is out of bounds
    cpdef bool is_colliding_with_stack(self, vector[vector[PieceType]] stack):
        cdef int x, y
        for x, y in self.get_cells():
            if stack[y][x] != PieceType.EMPTY:
                return True
        return False

    cpdef bool is_colliding_or_out_of_bounds(self, vector[vector[PieceType]] stack):
        return self.is_out_of_bounds(stack[0].size(), stack.size()) or self.is_colliding_with_stack(stack)

    cpdef place_on_stack(self, vector[vector[PieceType]] stack):
        cdef int x, y
        for x, y in self.get_cells():
            stack[y][x] = self.type

    cpdef bool have_same_cells(self, Piece other):
        return self.get_cells() == other.get_cells()

    def __hash__ (self) -> int:
        return hash((self.type, self.top_left, self.rotation))

    def __eq__(self, other: 'Piece') -> bool:
        return self.type == other.type and self.top_left == other.top_left and self.rotation == other.rotation
