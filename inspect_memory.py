import pickle
import player

with open("memory_using_before_clearing.pickle", 'rb') as f:
    memory = pickle.load(f)
    print("len(memory):", len(memory))