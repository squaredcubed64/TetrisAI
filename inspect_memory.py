import pickle

with open("linear_regression.pickle", 'rb') as f:
    memory = pickle.load(f)
    print("len(memory):", len(memory))