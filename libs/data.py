import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from importlib import import_module

def get_dataset(name, datasets_folder='datasets', seed=0, test_ratio=0.2):
    loader = import_module(f'.{name}.load', package=f'{datasets_folder}')
    X, Y = loader.load()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=test_ratio, random_state=seed)
    X_train, X_test, Y_train, Y_test = X[idx_train], X[idx_test], Y[idx_train], Y[idx_test]
    return X_train, X_test, Y_train, Y_test

def standardization_data(train, test, rel_indices):
    sx = StandardScaler()
    sx.fit(train[rel_indices])
    train = sx.transform(train)
    test = sx.transform(test)
    return train, test
