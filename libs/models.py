import torch
import torch.nn as nn

class NNetModel(nn.Module):
    def __init__(self, in_shape:int=1, out_shape:int=1, hidden_size=0, dropout:float=0.):
        super().__init__()
        layers = []
        last_size = in_shape
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] if hidden_size > 0 else []
        try:
            for size in iter(hidden_size):
                layers.append(nn.Linear(last_size, size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                last_size = size
        finally:
            layers.append(nn.Linear(last_size, out_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class CovarianceModel(nn.Module):
    def __init__(self, in_shape=1, out_shape=1, hidden_size=0, dropout=0):
        super().__init__()
        self.out_shape = out_shape
        self.model = NNetModel(in_shape=in_shape, out_shape=out_shape ** 2, hidden_size=hidden_size, dropout=dropout)

    def forward(self, X):
        output = self.model(X).reshape((-1, self.out_shape, self.out_shape))
        return torch.bmm(output.transpose(-2, -1), output)
