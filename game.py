import random
from typing import List
from piece import Piece
from piece_type import PieceType

class Game:
    SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS = 550, 660
    BOARD_WIDTH_PIXELS, BOARD_HEIGHT_PIXELS = 300, 660
    BOARD_WIDTH_CELLS, BOARD_HEIGHT_CELLS = 10, 22
    CELL_HEIGHT_PIXELS = BOARD_HEIGHT_PIXELS // BOARD_HEIGHT_CELLS
    CELL_WIDTH_PIXELS = BOARD_WIDTH_PIXELS // BOARD_WIDTH_CELLS

    FPS = 60
    FRAMES_PER_DROP = 30
    FRAMES_BETWEEN_ACTIONS = 5

    def __init__(self):
        self.stack = [[PieceType.EMPTY for _ in range(self.BOARD_WIDTH_CELLS)] for _ in range(self.BOARD_HEIGHT_CELLS)]
        self.current_piece = None
        self.next_piece_types = [random.choice([PieceType.I, PieceType.J, PieceType.L, PieceType.O, PieceType.S, PieceType.Z, PieceType.T]) for _ in range(5)]

    def clear_rows_and_return_rows_cleared(self) -> int:
        rows_to_clear = []
        for y in range(self.BOARD_HEIGHT_CELLS):
            if all(cell != PieceType.EMPTY for cell in self.stack[y]):
                rows_to_clear.append(y)
        for row in rows_to_clear:
            self.stack.pop(row)
            self.stack.insert(0, [PieceType.EMPTY for _ in range(self.BOARD_WIDTH_CELLS)])
        return len(rows_to_clear)

    def spawn_piece(self) -> None:
        next_piece_type = self.next_piece_types.pop(0)
        if next_piece_type == PieceType.O:
            self.current_piece = Piece(next_piece_type, (self.BOARD_WIDTH_CELLS // 2 - 1, 0))
        else:
            self.current_piece = Piece(next_piece_type, (self.BOARD_WIDTH_CELLS // 2 - 2, 0))
        
        self.next_piece_types.append(random.choice([PieceType.I, PieceType.J, PieceType.L, PieceType.O, PieceType.S, PieceType.Z, PieceType.T]))
    
    def update_stack_and_return_rows_cleared(self, stack: List[List[PieceType]]) -> int:
        self.stack = stack
        rows_cleared = self.clear_rows_and_return_rows_cleared()
        self.spawn_piece()
        return rows_cleared