import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.neighbors import KNeighborsRegressor

import torch

from os.path import isfile
from copy import deepcopy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BaseLearner:
    def __init__(self):
        self.scale = 1.
        self.cov = None

    def fit(self, X, y, to_force=False, **kwargs):
        self.cov = np.cov(y.T)
        return self

    def predict(self, X):
        return self.cov

    def __call__(self, X):
        return self.scale * self.predict(X)

    def score(self, X, y):
        covs = self.predict(X)
        if covs.ndim == 2:
            return np.einsum('...i,ij,...j->...', y, np.linalg.inv(covs), y) / self.scale
        return np.einsum('...i,...ij,...j->...', y, np.linalg.inv(covs), y) / self.scale

    def calibrate(self, X, y, q=0.9, reset=False):
        if reset:
            self.scale = 1.
        n = X.shape[0]
        scores = self.score(X, y)
        q = min(np.floor((n + 1) * q) / n, 1.)
        self.scale *= np.quantile(scores, q)
        return self

class NearestNeighborsLearner(BaseLearner):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.out_shape = 0
        self.ys = None

    def fit(self, X, y, **kwargs):
        super().fit(X, y)
        self.out_shape = y.shape[-1]
        y = np.einsum('...i,...j->...ij', y, y).reshape((-1, self.out_shape**2))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        covs = self.model.predict(X).reshape(-1, self.out_shape, self.out_shape)
        return 0.95 * covs + 0.05 * super().predict(None)

    @staticmethod
    def softmax(dist):
        return scipy_softmax(dist, axis=1)

class SGDLearner(BaseLearner):
    def __init__(self, model, optimizer_class, loss_func, reg_func=None, reg=0.):
        super().__init__()
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.loss_func = loss_func
        self.reg_func = reg_func
        self.reg = reg

    def fit(self, X, y, epochs:int=1000, batch_size:int=-1, early_stop:bool=True, to_print=True, **kwargs):
        n = X.shape[0]
        if n == 0:
            return self
        shuffle_idx = np.arange(n)
        if batch_size < 1 or batch_size >= n:
            batch_size = n
        else:
            np.random.shuffle(shuffle_idx)
        optimizer = self.optimizer_class(self.model.parameters())
        X = self.from_numpy(X)
        y = self.from_numpy(y)
        model = deepcopy(self.model)
        self.model.train()
        prev_loss = np.infty
        for ei in range(epochs):
            curr_loss = 0
            for idx in range(0, n, batch_size):
                optimizer.zero_grad()
                batch_x = X[shuffle_idx[idx: min(idx + batch_size, n)]]
                batch_y = y[shuffle_idx[idx: min(idx + batch_size, n)]]
                batch_pred = self.model(batch_x)
                loss = self.loss_func(batch_pred, batch_y)
                if callable(self.reg_func):
                    loss = self.reg_func(loss, batch_x)
                loss.backward()
                optimizer.step()
                curr_loss += self.to_numpy(loss) * min(batch_size, n - idx) / n
            if ei % 100 == 0 or ei == epochs-1:
                if to_print:
                    print(f'epoch: {ei}/{epochs}', f'loss: {curr_loss:.06f}', sep=', ', flush=True)
                if np.isnan(curr_loss):
                    self.model = model.to(device)
                    self.model.eval()
                    break
                if early_stop and np.isclose(curr_loss, prev_loss):
                    break
                model = deepcopy(self.model)
                prev_loss = curr_loss
        return self

    def predict(self, X):
        with torch.no_grad():
            self.model.eval()
            covs = self.to_numpy(self.model(self.from_numpy(X)))
        if self.reg > 0:
            covs += self.reg * np.eye(covs.shape[-1])
        return covs

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if not isfile(path):
            raise FileNotFoundError
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    @staticmethod
    def from_numpy(X):
        return torch.from_numpy(X).float().requires_grad_(False).to(device)

    @staticmethod
    def to_numpy(X):
        return X.detach().cpu().numpy()


