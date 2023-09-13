import numpy as np
import os

def load():
    absolute_path = os.path.dirname(__file__)
    data = np.load(f'{absolute_path}/rssi.npz')
    X = data['x']
    Y = data['y']
    return X, Y
