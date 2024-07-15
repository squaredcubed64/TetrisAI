import pickle
from player import Player

def inspect_memory(file_path: str) -> None:
    with open(file_path, 'rb') as file:
        memory = pickle.load(file)
    print(memory)

def inspect_weights(file_path: str, architecture: str) -> None:
    player = Player(architecture)
    player.load_model(file_path)
    for layer in player.model.layers:
        print(layer.get_weights())

inspect_memory("memory_dense.pickle")