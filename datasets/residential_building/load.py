import numpy as np
import pandas as pd
import os
import json

def load():
    absolute_path = os.path.dirname(__file__)
    data = pd.read_csv(f'{absolute_path}/data.csv', sep="|")
    with open(f'{absolute_path}/config.json') as f:
        config = json.load(f)
    X = data[config["continuous_variables"]].values
    Y = data[config["targets"]].values / 100
    return X, Y
