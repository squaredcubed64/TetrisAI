import copy
import time
from typing import Dict, List, Set, Tuple
from action import Action
from game import Game
from player import Player
from piece import Piece
import cProfile
import pstats
import random
import matplotlib.pyplot as plt

NUM_EPISODES = 8192
EPISODES_BETWEEN_SAVES = 4
EPISODES_BETWEEN_PLOTS = 4
architecture = "linear_regression"
player = Player(architecture)
model_load_path = None # architecture + ".keras"
model_save_path = None # architecture + ".keras"
memory_load_path = None # architecture + ".pickle"
memory_save_path = architecture + ".pickle"
rows_cleared_memory: List[int] = []

random.seed(42)

# @profile
def include_pieces_and_paths_dfs(piece: Piece, path: List[Action], stack: List[List[int]], terminal_piece_to_path: Dict[Piece, List[Action]], nonterminal_piece_to_path: Dict[Piece, List[Action]]) -> None:
    if piece in nonterminal_piece_to_path:
        return
    nonterminal_piece_to_path[piece] = path

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


        # Forced DROP in between each action
        new_piece.move((0, 1))
        new_path = path + [action, Action.DROP]
        if new_piece.is_colliding_or_out_of_bounds(stack):
            new_piece.move((0, -1))
            if not new_piece.is_colliding_or_out_of_bounds(stack):
                terminal_piece_to_path[new_piece] = new_path
        else:
            include_pieces_and_paths_dfs(new_piece, new_path, stack, terminal_piece_to_path, nonterminal_piece_to_path)

def calculate_results_and_paths(initial_stack: List[List[int]], initial_piece: Piece) -> List[Tuple[List[List[int]], List[Action]]]:
    terminal_piece_to_path: Dict[Piece, List[Action]] = {}
    nonterminal_piece_to_path: Dict[Piece, List[Action]] = {}
    include_pieces_and_paths_dfs(initial_piece, [], initial_stack, terminal_piece_to_path, nonterminal_piece_to_path)

    results_and_paths: List[Tuple[List[List[int]], List[Action]]] = []
    pieces_seen: List[Piece] = []
    for terminal_piece, path in terminal_piece_to_path.items():
        seen_this_piece = False
        for i in range(len(pieces_seen)):
            if terminal_piece.has_same_cells(pieces_seen[i]):
                seen_this_piece = True
                break
        
        if not seen_this_piece:
            stack_copy = copy.deepcopy(initial_stack)
            terminal_piece.place_on_stack(stack_copy)
            pieces_seen.append(terminal_piece)
            results_and_paths.append((stack_copy, path))

    return results_and_paths

def main():
    if model_load_path is not None:
        print("Loading model from", model_load_path)
        player.load_model(model_load_path)
    if memory_load_path is not None:
        print("Loading memory from", memory_load_path)
        player.load_memory(memory_load_path)

    for episode_number in range(NUM_EPISODES):
        game = Game()
        print("Episode", episode_number + 1, "of", NUM_EPISODES)
        total_rows_cleared = 0

        while True:
            initial_stack = copy.deepcopy(game.stack)
            results_and_paths = calculate_results_and_paths(game.stack, game.current_piece)
            if results_and_paths == []:
                player.memorize(initial_stack, None, 0)
                break

            best_stack = player.choose_state([stack for stack, _ in results_and_paths])
            rows_cleared = game.update_stack_and_return_rows_cleared(best_stack)
            total_rows_cleared += rows_cleared
            new_stack = copy.deepcopy(game.stack)
            player.memorize(initial_stack, new_stack, rows_cleared)
        
        print("Rows cleared:", total_rows_cleared)
        rows_cleared_memory.append(total_rows_cleared)
        player.try_to_fit_on_memory()
        player.update_epsilon(episode_number)

        if architecture == "linear_regression":
            for layer in player.model.layers:
                weights = layer.get_weights()  # returns a list of all weight tensors in the layer
                print("Weights:", [weights[0][i][0] for i in range(player.NUM_FEATURES)])
        
        if episode_number % EPISODES_BETWEEN_SAVES == EPISODES_BETWEEN_SAVES - 1:
            if model_save_path is not None:
                print("Saving model to", model_save_path)
                player.save_model(model_save_path)
            if memory_save_path is not None:
                print("Saving memory to", memory_save_path)
                player.save_memory(memory_save_path)
        
        if episode_number % EPISODES_BETWEEN_PLOTS == EPISODES_BETWEEN_PLOTS - 1:
            plt.plot(rows_cleared_memory, color="blue")
            plt.xlabel("Episode")
            plt.ylabel("Rows cleared")
            plt.ylim(bottom=0)
            plt.show(block=False)
            plt.pause(0.1)

def time_main():
    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

main()

# with cProfile.Profile() as profile:
#     main()

# results = pstats.Stats(profile)
# results.strip_dirs()
# results.dump_stats("train_player.prof")
