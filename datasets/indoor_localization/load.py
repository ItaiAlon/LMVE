import pandas as pd
import math
import os


def normalize(x, xmin, xmax, a, b):
    numerator = x - xmin
    denominator = xmax - xmin
    multiplier = b - a
    ans = (numerator / denominator) * multiplier + a
    return ans


sig_min = -104
sig_max = 0
tar_min = 0.25
tar_max = 1.0
no_sig = 100


def normalize_wifi(num):
    ans = 0
    num = float(num)
    if math.isclose(num, no_sig, rel_tol=1e-3):
        return 0
    else:
        ans = normalize(num, sig_min, sig_max, tar_min, tar_max)
        return ans


lat_min = 4864745.7450159714
lat_max = 4865017.3646842018
lat_tarmin = 0
lat_tarmax = 1


def normalize_lat(num):
    num = float(num)
    ans = normalize(num, lat_min, lat_max, lat_tarmin, lat_tarmax)
    return ans


long_min = -7695.9387549299299000
long_max = -7299.786516730871000
long_tarmin = 0
long_tarmax = 1


def normalize_long(num):
    num = float(num)
    ans = normalize(num, long_min, long_max, long_tarmin, long_tarmax)
    return ans


def load():
    absolute_path = os.path.dirname(__file__)
    df = pd.read_csv(f'{absolute_path}/TrainingData.csv')
    # load more examples:
    # df_val = pd.read_csv(f'{absolute_path}/ValidationData.csv')
    # df = pd.concat([df, df_val])
    df.drop(columns=["RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"], inplace=True)

    wifi_cells = df.columns[:519]
    # for i in wifi_cells:
    #     df[i] = df[i].apply(normalize_wifi)
    # df["LATITUDE"] = df["LATITUDE"].apply(normalize_lat)
    # df["LONGITUDE"] = df["LONGITUDE"].apply(normalize_long)

    X = df[wifi_cells].to_numpy()
    # Y = df[["LATITUDE", "LONGITUDE", "BUILDINGID", "FLOOR"]].to_numpy()
    Y = df[["LATITUDE", "LONGITUDE"]].to_numpy() / 1000

    return X, Y
