import os
import pickle

def save_stubs(stubs_path, object):
    if not os.path.exists(os.path.dirname(stubs_path)):
        os.mkdir(os.path.dirname(stubs_path))
    
    if stubs_path is not None:
        with open(stubs_path, 'wb') as f:
            pickle.dump(object, f)

def read_stubs(read_from_stubs, stubs_path):
    if read_from_stubs and stubs_path is not None and os.path.exists(stubs_path):
        with open(stubs_path, 'rb') as f:
            object = pickle.load(f)
            return object
    return None
        