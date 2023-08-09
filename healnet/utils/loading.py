import pickle


def pickle_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def unpickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
