import pickle
from collections import deque

with open('memory_original_weights.pickle', 'rb') as file:
    memory_list = pickle.load(file)

memory_deque = deque(memory_list)

with open('memory_original_weights_deque.pickle', 'wb') as file:
    pickle.dump(memory_deque, file)