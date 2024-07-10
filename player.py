import random
from typing import List, Tuple
from keras import layers, models
import numpy as np
from game import Game, PieceType

class Player:
    def __init__(self, architecture: str) -> None:
        self.BATCH_SIZE = 512
        self.REPLAY_START = 2048
        self.DISCOUNT_FACTOR = 0.96
        self.NUM_EPOCHS = 1
        self.NUM_FEATURES_IN_DENSE_MODEL = 4

        self.epsilon = 1.0
        self.EPSILON_MIN = 0.005
        self.EPSILON_DECAY_END_EPISODE = 1024

        self.architecture = architecture

        # List of (state (binary grid), next_state (binary grid), reward) tuples
        self.memory: List[Tuple[List[List[int]], List[List[int]] | None, float]] = []
        # # List of (max_height, full_rows, bumpiness, holes) tuples
        # self.memory_dense: List[Tuple[int, int, int, int]] = []

        if architecture == "cnn":
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
        elif architecture == "dense":
            self.model = models.Sequential([
                layers.Dense(64, input_dim=self.NUM_FEATURES_IN_DENSE_MODEL, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def convert_stack_to_binary_grid(self, state: List[List[PieceType]]) -> List[List[int]]:
        return [[1 if cell != PieceType.EMPTY else 0 for cell in row] for row in state]

    def memorize(self, state: List[List[PieceType]], next_state: List[List[PieceType]] | None, reward: float) -> None:
        if next_state is None:
            next_state_as_binary_grid = None
        else:
            next_state_as_binary_grid = self.convert_stack_to_binary_grid(next_state)
        self.memory.append((self.convert_stack_to_binary_grid(state), next_state_as_binary_grid, reward))

        # if self.architecture == "dense":
        #     self.memory_dense.append(self.get_features(state))

    def get_height(self, stack: List[List[PieceType]], x: int) -> int:
        for y in range(Game.BOARD_HEIGHT_CELLS):
            if stack[y][x] != PieceType.EMPTY:
                return Game.BOARD_HEIGHT_CELLS - y
        return 0

    def get_max_height(self, stack: List[List[PieceType]]) -> int:
        return max([self.get_height(stack, x) for x in range(Game.BOARD_WIDTH_CELLS)])

    def get_full_rows(self, stack: List[List[PieceType]]) -> int:
        full_rows = 0
        for y in range(Game.BOARD_HEIGHT_CELLS):
            if all(cell != PieceType.EMPTY for cell in stack[y]):
                full_rows += 1
        return full_rows
        
    def get_bumpiness(self, stack: List[List[PieceType]]) -> int:
        heights = [self.get_height(stack, x) for x in range(Game.BOARD_WIDTH_CELLS)]
        return sum([(heights[i] - heights[i + 1]) ** 2 for i in range(Game.BOARD_WIDTH_CELLS - 1)])
    
    # Returns the number of empty cells trapped underneath a filled cell
    def get_holes(self, stack: List[List[PieceType]]) -> int:
        holes = 0
        for x in range(Game.BOARD_WIDTH_CELLS):
            filled_cell_found = False
            for y in range(Game.BOARD_HEIGHT_CELLS):
                if stack[y][x] != PieceType.EMPTY:
                    filled_cell_found = True
                elif filled_cell_found:
                    holes += 1
        return holes
    
    def get_features(self, stack: List[List[PieceType]]) -> Tuple[int, int, int, int]:
        return (self.get_max_height(stack), self.get_full_rows(stack), self.get_bumpiness(stack), self.get_holes(stack))

    def get_best_state(self, states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
        best_state = None
        best_value = None

        for state in states:
            if self.architecture == "cnn":
                binary_grid = self.convert_stack_to_binary_grid(state)
                value = self.model.predict(np.reshape(binary_grid, (1, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)), verbose=0)[0][0]
            elif self.architecture == "dense":
                value = self.model.predict(np.reshape(self.get_features(state), (1, self.NUM_FEATURES_IN_DENSE_MODEL)), verbose=0)[0][0]

            if best_value is None or value > best_value:
                best_state = state
                best_value = value
        
        return best_state

    def choose_state(self, states: List[List[List[PieceType]]]) -> List[List[PieceType]]:
        if np.random.rand() < self.epsilon:
            return states[np.random.randint(len(states))]
        else:
            return self.get_best_state(states)
        
    def try_to_fit_on_memory(self) -> None:
        if len(self.memory) < self.REPLAY_START:
            print("Not enough samples in memory to fit the model. Skipping training.")
            return
        else:
            print("Fitting to memory")

        batch = random.sample(self.memory, self.BATCH_SIZE)

        batch_without_terminal_transitions = list(filter(lambda transition : transition[1] is not None, batch))

        if self.architecture == "cnn":
            nonterminal_next_states = np.reshape([next_state for _, next_state, _  in batch_without_terminal_transitions],
                                                 (len(batch_without_terminal_transitions), Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1))
        elif self.architecture == "dense":
            nonterminal_next_states = np.array([self.get_features(next_state) for _, next_state, _ in batch_without_terminal_transitions])
        nonterminal_next_q_values = np.array([s[0] for s in self.model.predict(nonterminal_next_states, verbose=0)])
    
        next_q_values = []
        nonterminal_index = 0
        for step_number in range(self.BATCH_SIZE):
            if batch[step_number][1] is None:
                next_q_values.append(0)
            else:
                next_q_values.append(nonterminal_next_q_values[nonterminal_index])
                nonterminal_index += 1

        x = []
        y = []

        for i, (state, _, reward) in enumerate(batch):
            q_value = reward + self.DISCOUNT_FACTOR * next_q_values[i]

            x.append(state)
            y.append(q_value)

        if self.architecture == "cnn":
            self.model.fit(np.reshape(x, (self.BATCH_SIZE, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)), np.array(y), epochs=self.NUM_EPOCHS, verbose=0)
        elif self.architecture == "dense":
            self.model.fit(np.array([self.get_features(state) for state in x]), np.array(y), epochs=self.NUM_EPOCHS, verbose=0)
    
    def update_epsilon(self, episode_number: int) -> None:
        if episode_number < self.EPSILON_DECAY_END_EPISODE:
            self.epsilon = 1.0 - (1.0 - self.EPSILON_MIN) * (episode_number / self.EPSILON_DECAY_END_EPISODE)
        else:
            self.epsilon = self.EPSILON_MIN
    
    def save_model(self, path: str) -> None:
        self.model.save(path)
    
    def load_model(self, path: str) -> None:
        self.model = models.load_model(path)