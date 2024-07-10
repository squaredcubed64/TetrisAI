import random
from typing import List, Tuple
from keras import layers, models
import numpy as np
from game import Game, PieceType

class Player:
    def __init__(self) -> None:
        self.BATCH_SIZE = 512
        self.REPLAY_START = 2048
        self.DISCOUNT_FACTOR = 0.96
        self.NUM_EPOCHS = 1

        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.005
        self.EPSILON_DECAY_END_EPISODE = 1024
        self.EPSILON_DECAY = (self.EPSILON - self.EPSILON_MIN) / self.EPSILON_DECAY_END_EPISODE

        # List of (state (binary grid), next_state (binary grid), reward, terminal) tuples
        self.memory: List[Tuple[List[List[int]], List[List[int]], float, bool]] = []

        # TODO Try removing the last conv2d and maxpooling2d layers to increase capacity
        self.model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def convert_stack_to_binary_grid(self, state: List[List[PieceType]]) -> List[List[int]]:
        return [[1 if cell != PieceType.EMPTY else 0 for cell in row] for row in state]

    def memory_append(self, state: List[List[PieceType]], next_state: List[List[PieceType]], reward: float, terminal: bool) -> None:
        self.memory.append((self.convert_stack_to_binary_grid(state), self.convert_stack_to_binary_grid(next_state), reward, terminal))

    def get_best_state(self, states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
        best_state = None
        best_value = None

        for state in states:
            binary_grid = self.convert_stack_to_binary_grid(state)
            value = self.model.predict(np.reshape(binary_grid, (1, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)))[0][0]

            if best_value is None or value > best_value:
                best_state = state
                best_value = value
        
        return best_state

    def choose_state(self, states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
        if np.random.rand() < self.EPSILON:
            return states[np.random.randint(len(states))]
        else:
            return self.get_best_state(states)
        
    def try_to_fit_on_memory(self) -> None:
        if len(self.memory) < self.BATCH_SIZE:
            raise ValueError("Not enough samples in memory to fit the model.")

        batch = random.sample(self.memory, self.BATCH_SIZE)

        next_states = np.reshape([next_state for _, next_state, _, _ in batch], (self.BATCH_SIZE, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1))
        next_q_values = np.array([s[0] for s in self.model.predict(next_states)])

        x = []
        y = []

        for i, (state, next_state, reward, terminal) in enumerate(batch):
            q_value = reward
            if not terminal:
                q_value += self.DISCOUNT_FACTOR * next_q_values[i]

            x.append(state)
            y.append(q_value)

        self.model.fit(np.reshape(x, (self.BATCH_SIZE, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)), np.array(y), epochs=self.NUM_EPOCHS, verbose=0)
    
    def try_to_decay_epsilon(self) -> None:
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON -= self.EPSILON_DECAY