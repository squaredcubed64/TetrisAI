import pygame
import random
from enum import Enum, auto
from typing import List, Set, Tuple, Dict
from functools import lru_cache
import copy

pygame.init()

SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS = 550, 660
BOARD_WIDTH_PIXELS, BOARD_HEIGHT_PIXELS = 300, 660
BOARD_WIDTH_CELLS, BOARD_HEIGHT_CELLS = 10, 22
CELL_HEIGHT_PIXELS = BOARD_HEIGHT_PIXELS // BOARD_HEIGHT_CELLS
CELL_WIDTH_PIXELS = BOARD_WIDTH_PIXELS // BOARD_WIDTH_CELLS

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BLUE = (0, 255, 255)
DARK_BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 128, 0)
RED = (128, 0, 0)
MAGENTA = (165, 0, 165)

FPS = 60
FRAMES_PER_DROP = 30
FRAMES_BETWEEN_ACTIONS = 5
clock = pygame.time.Clock()

screen = pygame.display.set_mode((SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS))

# source for rotations: https://tetris.wiki/Super_Rotation_System
class Rotation(Enum):
    ZERO = 0
    RIGHT = 1
    TWO = 2
    LEFT = 3

    def clockwise(self) -> 'Rotation':
        return Rotation((self.value + 1) % 4)

    def counterclockwise(self) -> 'Rotation':
        return Rotation((self.value - 1) % 4)

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

class Piece:
    def __init__(self, type: PieceType, top_left: Tuple[int, int], rotation: Rotation = Rotation.ZERO):
        self.type = type
        self.top_left = top_left
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
    
    def is_out_of_bounds(self) -> bool:
        for x, y in self.get_cells():
            if x < 0 or x >= BOARD_WIDTH_CELLS or y < 0 or y >= BOARD_HEIGHT_CELLS:
                return True
        return False

    # indexError if piece is out of bounds
    def is_colliding_with_stack(self, stack: List[List[PieceType]]) -> bool:
        for x, y in self.get_cells():
            if stack[y][x] != PieceType.EMPTY:
                return True
        return False
    
    def is_colliding_or_out_of_bounds(self, stack: List[List[PieceType]]) -> bool:
        return self.is_out_of_bounds() or self.is_colliding_with_stack(stack)
    
    def place_on_stack(self, stack: List[List[PieceType]]) -> None:
        for x, y in self.get_cells():
            stack[y][x] = self.type

    def __hash__ (self) -> int:
        return hash((self.type, self.top_left, self.rotation))

    def __eq__(self, other: 'Piece') -> bool:
        return self.type == other.type and self.top_left == other.top_left and self.rotation == other.rotation

class Action(Enum):
    DO_NOTHING = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    ROTATE_CLOCKWISE = auto()
    ROTATE_COUNTERCLOCKWISE = auto()
    DROP = auto()

def include_pieces_and_paths_dfs(piece: Piece, path: List[Action], stack: List[List[PieceType]], terminal_piece_to_path: Dict[Piece, List[Action]], non_terminal_piece_to_path: Dict[Piece, List[Action]]) -> None:
    if piece in non_terminal_piece_to_path:
        return
    non_terminal_piece_to_path[piece] = path

    for action in [Action.DO_NOTHING, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.ROTATE_CLOCKWISE, Action.ROTATE_COUNTERCLOCKWISE]:
        if action == Action.DO_NOTHING:
            new_piece = Piece(piece.type, piece.top_left, piece.rotation)
        elif action == Action.MOVE_LEFT:
            new_piece = Piece(piece.type, (piece.top_left[0] - 1, piece.top_left[1]), piece.rotation)
            if new_piece.is_colliding_or_out_of_bounds(stack):
                new_piece.move((1, 0))
        elif action == Action.MOVE_RIGHT:
            new_piece = Piece(piece.type, (piece.top_left[0] + 1, piece.top_left[1]), piece.rotation)
            if new_piece.is_colliding_or_out_of_bounds(stack):
                new_piece.move((-1, 0))
        elif action == Action.ROTATE_COUNTERCLOCKWISE:
            new_piece = Piece(piece.type, piece.top_left, piece.rotation.counterclockwise())

            rotation_successful = False
            for offset in new_piece.type.wall_kicks[new_piece.rotation.clockwise()][new_piece.rotation]:
                new_piece.move(offset)
                if not new_piece.is_colliding_or_out_of_bounds(stack):
                    rotation_successful = True
                    break
                new_piece.move((-offset[0], -offset[1]))

            if not rotation_successful:
                new_piece.rotate_clockwise()
        elif action == Action.ROTATE_CLOCKWISE:
            new_piece = Piece(piece.type, piece.top_left, piece.rotation.clockwise())

            rotation_successful = False
            for offset in new_piece.type.wall_kicks[new_piece.rotation.counterclockwise()][new_piece.rotation]:
                new_piece.move(offset)
                if not new_piece.is_colliding_or_out_of_bounds(stack):
                    rotation_successful = True
                    break
                new_piece.move((-offset[0], -offset[1]))

            if not rotation_successful:
                new_piece.rotate_counterclockwise()

        # Forced DROP in between each action
        new_piece.move((0, 1))
        new_path = path + [action, Action.DROP]
        if new_piece.is_colliding_or_out_of_bounds(stack):
            new_piece.move((0, -1))
            terminal_piece_to_path[new_piece] = new_path
        else:
            include_pieces_and_paths_dfs(new_piece, new_path, stack, terminal_piece_to_path, non_terminal_piece_to_path)

def calculate_results_and_paths(initial_stack: List[List[PieceType]], initial_piece) -> List[Tuple[List[List[PieceType]], List[Action]]]:
    terminal_piece_to_path = {}
    non_terminal_piece_to_path = {}
    include_pieces_and_paths_dfs(initial_piece, [], initial_stack, terminal_piece_to_path, non_terminal_piece_to_path)

    results_and_paths = []
    for terminal_piece, path in terminal_piece_to_path.items():
        stack_copy = copy.deepcopy(initial_stack)
        terminal_piece.place_on_stack(stack_copy)
        results_and_paths.append((stack_copy, path))
    
    return results_and_paths

def num_holes(stack: List[List[PieceType]]) -> int:
    holes = 0
    for x in range(BOARD_WIDTH_CELLS):
        hole_found = False
        for y in range(BOARD_HEIGHT_CELLS):
            if stack[y][x] != PieceType.EMPTY:
                hole_found = True
            elif hole_found:
                holes += 1
    return holes

def evaluate_stack(stack: List[List[PieceType]]) -> float:
    holes = 0
    for x in range(BOARD_WIDTH_CELLS):
        hole_found = False
        for y in range(BOARD_HEIGHT_CELLS):
            if stack[y][x] != PieceType.EMPTY:
                hole_found = True
            elif hole_found:
                holes += 1
    return holes

main_stack = [[PieceType.EMPTY for _ in range(BOARD_WIDTH_CELLS)] for _ in range(BOARD_HEIGHT_CELLS)]
current_piece = None
next_piece_types = [random.choice([PieceType.I, PieceType.J, PieceType.L, PieceType.O, PieceType.S, PieceType.Z, PieceType.T]) for _ in range(5)]
frames_since_last_drop = 0
frames_since_last_action = 0
rows_cleared = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    def clear_rows() -> None:
        global rows_cleared
        rows_to_clear = []
        for y in range(BOARD_HEIGHT_CELLS):
            if all(cell != PieceType.EMPTY for cell in main_stack[y]):
                rows_to_clear.append(y)
        for row in rows_to_clear:
            main_stack.pop(row)
            main_stack.insert(0, [PieceType.EMPTY for _ in range(BOARD_WIDTH_CELLS)])
        rows_cleared += len(rows_to_clear)

    def spawn_piece(piece_type: PieceType) -> None:
        global current_piece
        if piece_type == PieceType.O:
            current_piece = Piece(piece_type, (BOARD_WIDTH_CELLS // 2 - 1, 0))
        else:
            current_piece = Piece(piece_type, (BOARD_WIDTH_CELLS // 2 - 2, 0))

    def fill_cell(x: int, y: int, color: Tuple[int, int, int]) -> None:
        pygame.draw.rect(screen, color, (x * CELL_WIDTH_PIXELS, y * CELL_HEIGHT_PIXELS, CELL_WIDTH_PIXELS, CELL_HEIGHT_PIXELS))

    def draw_stack() -> None:
        for x in range(BOARD_WIDTH_CELLS):
            for y in range(BOARD_HEIGHT_CELLS):
                fill_cell(x, y, main_stack[y][x].color)

    def draw_current_piece() -> None:
        if current_piece is not None:
            for x, y in current_piece.get_cells():
                fill_cell(x, y, current_piece.type.color)

    def draw_grid() -> None:
        for x in range(0, BOARD_WIDTH_PIXELS + 1, CELL_WIDTH_PIXELS):
            pygame.draw.line(screen, WHITE, (x, 0), (x, BOARD_HEIGHT_PIXELS))
        for y in range(0, BOARD_HEIGHT_PIXELS, CELL_HEIGHT_PIXELS):
            pygame.draw.line(screen, WHITE, (0, y), (BOARD_WIDTH_PIXELS, y))
        pygame.draw.line(screen, WHITE, (0, BOARD_HEIGHT_PIXELS - 1), (BOARD_WIDTH_PIXELS, BOARD_HEIGHT_PIXELS - 1))

    def draw_score() -> None:
        font = pygame.font.Font(None, 36)
        text = font.render("Lines Cleared: " + str(rows_cleared), True, WHITE)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH_PIXELS - 120, 50)
        screen.blit(text, text_rect)
    
    def draw_next_pieces() -> None:
        font = pygame.font.Font(None, 36)
        text = font.render("Next Pieces:", True, WHITE)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH_PIXELS - 120, 100)
        screen.blit(text, text_rect)
        for i, piece in enumerate(next_piece_types):
            for x, y in piece.up_body:
                fill_cell(x + BOARD_WIDTH_CELLS + 3, y + i * 3 + 5, piece.color)

    keys = pygame.key.get_pressed()
    if frames_since_last_action > FRAMES_BETWEEN_ACTIONS:
        if current_piece is not None:
            if keys[pygame.K_LEFT]:
                current_piece.move((-1, 0))
                if current_piece.is_colliding_or_out_of_bounds(main_stack):
                    current_piece.move((1, 0))
                frames_since_last_action = 0
            elif keys[pygame.K_RIGHT]:
                current_piece.move((1, 0))
                if current_piece.is_colliding_or_out_of_bounds(main_stack):
                    current_piece.move((-1, 0))
                frames_since_last_action = 0
            elif keys[pygame.K_z]:
                current_piece.rotate_counterclockwise()
                rotation_successful = False

                for offset in current_piece.type.wall_kicks[current_piece.rotation][current_piece.rotation.counterclockwise()]:
                    current_piece.move(offset)
                    if not current_piece.is_colliding_or_out_of_bounds(main_stack):
                        rotation_successful = True
                        break
                    current_piece.move((-offset[0], -offset[1]))

                if not rotation_successful:
                    current_piece.rotate_clockwise()
                frames_since_last_action = 0
            elif keys[pygame.K_x]:
                current_piece.rotate_clockwise()
                rotation_successful = False

                for offset in current_piece.type.wall_kicks[current_piece.rotation][current_piece.rotation.clockwise()]:
                    current_piece.move(offset)
                    if not current_piece.is_colliding_or_out_of_bounds(main_stack):
                        rotation_successful = True
                        break
                    current_piece.move((-offset[0], -offset[1]))
                
                if not rotation_successful:
                    current_piece.rotate_counterclockwise()
                frames_since_last_action = 0

    if frames_since_last_drop == FRAMES_PER_DROP:
        if current_piece is None:
            spawn_piece(next_piece_types.pop(0))
            next_piece_types.append(random.choice([PieceType.I, PieceType.J, PieceType.L, PieceType.O, PieceType.S, PieceType.Z, PieceType.T]))
        else:
            current_piece.move((0, 1))
            if current_piece.is_colliding_or_out_of_bounds(main_stack):
                current_piece.move((0, -1))
                if current_piece.is_colliding_or_out_of_bounds(main_stack):
                    running = False
                current_piece.place_on_stack(main_stack)
                current_piece = None
        frames_since_last_drop = 0

    clear_rows()

    screen.fill(BLACK)
    draw_stack()
    draw_current_piece()
    draw_grid()
    draw_score()
    draw_next_pieces()

    frames_since_last_drop += 1
    frames_since_last_action += 1

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()