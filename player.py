import random
from typing import List, Tuple
from keras import layers, models
import numpy as np
from game import BOARD_HEIGHT_CELLS, BOARD_WIDTH_CELLS, PieceType

EPSILON = 0.1
BATCH_SIZE = 512
DISCOUNT_FACTOR = 0.99
NUM_EPOCHS = 1

# List of (state (binary grid), next_state (binary grid), reward, terminal) tuples
memory: List[Tuple[List[List[int]], List[List[int]], float, bool]] = []

# TODO Try removing the last conv2d and maxpooling2d layers to increase capacity
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(BOARD_HEIGHT_CELLS, BOARD_WIDTH_CELLS, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

def convert_stack_to_binary_grid(state: List[List[PieceType]]) -> List[List[int]]:
    return [[1 if cell != PieceType.EMPTY else 0 for cell in row] for row in state]

def memory_append(state: List[List[PieceType]], next_state: List[List[PieceType]], reward: float, terminal: bool) -> None:
    memory.append((convert_stack_to_binary_grid(state), convert_stack_to_binary_grid(next_state), reward, terminal))

def get_best_state(states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
    best_state = None
    best_value = None

    for state in states:
        binary_grid = convert_stack_to_binary_grid(state)
        value = model.predict(np.reshape(binary_grid, (1, BOARD_HEIGHT_CELLS, BOARD_WIDTH_CELLS, 1)))[0][0]

        if best_value is None or value > best_value:
            best_state = state
            best_value = value
    
    return best_state

def choose_state(states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
    if np.random.rand() < EPSILON:
        return states[np.random.randint(len(states))]
    else:
        return get_best_state(states)
    
def fit_using_memory() -> None:
    if len(memory) < BATCH_SIZE:
        raise ValueError("Not enough samples in memory to fit the model.")

    batch = random.sample(memory, BATCH_SIZE)

    next_states = np.reshape([next_state for _, next_state, _, _ in batch], (BATCH_SIZE, BOARD_HEIGHT_CELLS, BOARD_WIDTH_CELLS, 1))
    next_q_values = np.array([s[0] for s in model.predict(next_states)])

    x = []
    y = []

    for i, (state, next_state, reward, terminal) in enumerate(batch):
        q_value = reward
        if not terminal:
            q_value += DISCOUNT_FACTOR * next_q_values[i]

        x.append(state)
        y.append(q_value)

    model.fit(np.reshape(x, (BATCH_SIZE, BOARD_HEIGHT_CELLS, BOARD_WIDTH_CELLS, 1)), np.array(y), epochs=NUM_EPOCHS, verbose=0)