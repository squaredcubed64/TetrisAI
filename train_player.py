import copy
from typing import Dict, List, Tuple
from action import Action
from game import Game, Piece, PieceType
from player import Player

NUM_EPISODES = 2048
player = Player()

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
            if not new_piece.is_colliding_or_out_of_bounds(stack):
                terminal_piece_to_path[new_piece] = new_path
        else:
            include_pieces_and_paths_dfs(new_piece, new_path, stack, terminal_piece_to_path, non_terminal_piece_to_path)

def stacks_are_equal(stack1: List[List[PieceType]], stack2: List[List[PieceType]]) -> bool:
    for row1, row2 in zip(stack1, stack2):
        for cell1, cell2 in zip(row1, row2):
            if (cell1 == PieceType.EMPTY) != (cell2 == PieceType.EMPTY):
                return False
    return True

def calculate_results_and_paths(initial_stack: List[List[PieceType]], initial_piece) -> List[Tuple[List[List[PieceType]], List[Action]]]:
    terminal_piece_to_path = {}
    non_terminal_piece_to_path = {}
    include_pieces_and_paths_dfs(initial_piece, [], initial_stack, terminal_piece_to_path, non_terminal_piece_to_path)

    results_and_paths = []
    for terminal_piece, path in terminal_piece_to_path.items():
        stack_copy = copy.deepcopy(initial_stack)
        terminal_piece.place_on_stack(stack_copy)

        seen_this_stack = False
        for i in range(len(results_and_paths)):
            if stacks_are_equal(results_and_paths[i][0], stack_copy):
                if len(path) < len(results_and_paths[i][1]):
                    results_and_paths[i] = (stack_copy, path)
                seen_this_stack = True
                break
        
        if not seen_this_stack:
            results_and_paths.append((stack_copy, path))

    return results_and_paths

for episode_number in range(NUM_EPISODES):
    game = Game()
    print("Episode", episode_number + 1, "of", NUM_EPISODES)
    total_rows_cleared = 0

    while True:
        initial_stack = copy.deepcopy(game.stack)
        results_and_paths = calculate_results_and_paths(game.stack, game.current_piece)
        if results_and_paths == []:
            player.memory_append(initial_stack, None, 0, True)
            break

        best_stack = player.choose_state([stack for stack, _ in results_and_paths])
        rows_cleared = game.update_stack_and_return_rows_cleared(best_stack)
        total_rows_cleared += rows_cleared
        new_stack = copy.deepcopy(game.stack)
        player.memory_append(initial_stack, new_stack, rows_cleared, False)
    
    print("Rows cleared:", total_rows_cleared)
    player.try_to_fit_on_memory()
    player.try_to_decay_epsilon()

# def debug_print_stack(stack: List[List[PieceType]]) -> None:
#     for row in stack:
#         for cell in row:
#             print(cell, end='')
#         print()

# results_and_paths = calculate_results_and_paths([[PieceType.EMPTY for _ in range(BOARD_WIDTH_CELLS)] for _ in range(BOARD_HEIGHT_CELLS)], Piece(PieceType.I, (0, 0)))
# for stack, path in results_and_paths:
#     print_stack(stack)
#     print(path)
#     print()

# def num_holes(stack: List[List[PieceType]]) -> int:
#     holes = 0
#     for x in range(BOARD_WIDTH_CELLS):
#         hole_found = False
#         for y in range(BOARD_HEIGHT_CELLS):
#             if stack[y][x] != PieceType.EMPTY:
#                 hole_found = True
#             elif hole_found:
#                 holes += 1
#     return holes
