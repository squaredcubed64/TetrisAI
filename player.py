import random
from typing import List, Tuple
from keras import layers, models, initializers
import numpy as np
from game import Game
import pickle

class Player:
    def __init__(self, architecture: str) -> None:
        self.BATCH_SIZE = 14879
        self.REPLAY_START = 2048
        self.DISCOUNT_FACTOR = .9
        self.NUM_EPOCHS = 16
        self.NUM_FEATURES = 4

        self.EPSILON_MAX = 0
        self.EPSILON_MIN = 0
        self.EPSILON_DECAY_END_EPISODE = 1024
        self.epsilon = 0
        
        self.architecture = architecture

        # List of (state (binary grid), next_state (binary grid), reward) tuples (after clearing)
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
                layers.Dense(64, input_dim=self.NUM_FEATURES, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
        elif architecture == "linear_regression":
            self.model = models.Sequential([
                layers.Dense(1, input_dim=self.NUM_FEATURES, activation='linear', kernel_initializer=initializers.RandomNormal(stddev=0.01))
            ])
            self.model.layers[0].set_weights([np.array([[0.01], [0.1], [-0.1], [-2]]), np.array([0])])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def memorize(self, state: List[List[int]], next_state: List[List[int]] | None, reward: float) -> None:
        self.memory.append((state, next_state, reward))

    def get_height(self, stack: List[List[int]], x: int) -> int:
        for y in range(Game.BOARD_HEIGHT_CELLS):
            if stack[y][x] == 1:
                return Game.BOARD_HEIGHT_CELLS - y
        return 0

    def get_max_height(self, stack_after_clearing: List[List[int]]) -> int:
        return max([self.get_height(stack_after_clearing, x) for x in range(Game.BOARD_WIDTH_CELLS)])
        
    def get_bumpiness(self, stack: List[List[int]]) -> int:
        heights = [self.get_height(stack, x) for x in range(Game.BOARD_WIDTH_CELLS)]
        return sum([(heights[i] - heights[i + 1]) ** 2 for i in range(Game.BOARD_WIDTH_CELLS - 1)])
    
    # Returns the number of empty cells trapped underneath a filled cell
    def get_holes(self, stack: List[List[int]]) -> int:
        holes = 0
        for x in range(Game.BOARD_WIDTH_CELLS):
            filled_cell_found = False
            for y in range(Game.BOARD_HEIGHT_CELLS):
                if stack[y][x] == 1:
                    filled_cell_found = True
                elif filled_cell_found:
                    holes += 1
        return holes
    
    # Returns None for full rows because after clearing, there will be no full rows. Thus, the user must insert a value for None
    def get_features_of_stack_after_clearing(self, stack_after_clearing: List[List[int]]) -> Tuple[int, None, int, int]:
        return (self.get_max_height(stack_after_clearing), None, self.get_bumpiness(stack_after_clearing), self.get_holes(stack_after_clearing))
    
    def get_features_of_stack_before_clearing(self, stack_before_clearing: List[List[int]]) -> Tuple[int, int, int, int]:
        rows_cleared = Game.clear_rows_and_return_rows_cleared(stack_before_clearing)
        return (self.get_max_height(stack_before_clearing), rows_cleared, self.get_bumpiness(stack_before_clearing), self.get_holes(stack_before_clearing))

    def get_best_state(self, states_before_clearing: List[List[List[int]]]) -> List[List[int]]:
        best_state = None
        best_value = None

        for state in states_before_clearing:
            if self.architecture == "cnn":
                value = self.model.predict(np.reshape(state, (1, Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1)), verbose=0)[0][0]
            elif self.architecture == "dense" or self.architecture == "linear_regression":
                value = self.model.predict(np.reshape(self.get_features_of_stack_before_clearing(state), (1, self.NUM_FEATURES)), verbose=0)[0][0]

            if best_value is None or value > best_value:
                best_state = state
                best_value = value
        
        return best_state

    def choose_state(self, states_before_clearing: List[List[List[int]]]) -> List[List[int]]:
        if random.random() < self.epsilon:
            return states_before_clearing[random.randint(0, len(states_before_clearing) - 1)]
        else:
            return self.get_best_state(states_before_clearing)
    
    def get_features_of_next_states_from_batch(self, batch: List[Tuple[List[List[int]], List[List[int]] | None, float]]) -> List[Tuple[int, int, int, int]]:
        features_without_rewards = [self.get_features_of_stack_after_clearing(next_state) for _, next_state, _ in batch]
        return [(features_without_rewards[i][0], batch[i][2], features_without_rewards[i][2], features_without_rewards[i][3]) for i in range(len(features_without_rewards))]

    def try_to_fit_on_memory(self) -> None:
        if len(self.memory) < self.REPLAY_START:
            print("Not enough samples in memory to fit the model. Skipping training.")
            return
        # TODO uncomment
        # else:
            # print("Fitting to memory")

        batch = random.sample(self.memory, self.BATCH_SIZE)

        batch_without_terminal_transitions = list(filter(lambda transition : transition[1] is not None, batch))

        if self.architecture == "cnn":
            nonterminal_next_states = np.reshape([next_state for _, next_state, _  in batch_without_terminal_transitions],
                                                 (len(batch_without_terminal_transitions), Game.BOARD_HEIGHT_CELLS, Game.BOARD_WIDTH_CELLS, 1))
        elif self.architecture == "dense" or self.architecture == "linear_regression":
            nonterminal_next_states = np.array(self.get_features_of_next_states_from_batch(batch_without_terminal_transitions))
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
        elif self.architecture == "dense" or self.architecture == "linear_regression":
            average_reward = sum([reward for _, _, reward in batch]) / len(batch)
            features = [self.get_features_of_stack_after_clearing(state) for state in x]
            for i in range(len(features)):
                features[i] = (features[i][0], average_reward, features[i][2], features[i][3])
            self.model.fit(np.array(features), np.array(y), epochs=self.NUM_EPOCHS, verbose=0)
            
    def update_epsilon(self, episode_number: int) -> None:
        if episode_number < self.EPSILON_DECAY_END_EPISODE:
            self.epsilon = self.EPSILON_MAX - (self.EPSILON_MAX - self.EPSILON_MIN) * (episode_number / self.EPSILON_DECAY_END_EPISODE)
        else:
            self.epsilon = self.EPSILON_MIN
    
    def save_model(self, path: str) -> None:
        self.model.save(path)
        # TODO remove
        self.model = models.load_model(path)
    
    def load_model(self, path: str) -> None:
        self.model = models.load_model(path)

    def save_memory(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def load_memory(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)