import copy
import random
from piece_type import PieceType
import copy
import random
from piece_type import PieceType
import time

def piece_type_test():
    grid = [[random.choice(list(PieceType)) for _ in range(10)] for _ in range(20)]
    start_time = time.time()
    for _ in range(100000):
        grid_copy = copy.deepcopy(grid)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

def binary_grid_test():
    binary_grid = [[random.choice([0, 1]) for _ in range(10)] for _ in range(20)]
    start_time = time.time()
    for _ in range(100000):
        binary_grid_copy = copy.deepcopy(binary_grid)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

piece_type_test()
binary_grid_test()