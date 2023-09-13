from libs.data import get_dataset, train_test_split, standardization_data
from libs.learners import BaseLearner, SGDLearner, NearestNeighborsLearner
from libs.losses import cov_gaussian_loss, cov_mse_loss
from libs.models import CovarianceModel

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import numpy as np
import torch
import pandas as pd

from functools import partial
import os
import time
import random

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def score(X, diff, estimator: BaseLearner, n_neighbors=1., cal_type='None', name='', data='', q=.9):
    assert n_neighbors > 0
    m, d = diff.shape

    scores = estimator.score(X, diff)
    cq = np.quantile(scores, q)

    volume = np.mean(np.sqrt(np.linalg.det(estimator(X))))
    normalized_volume = volume * cq ** (d/2)

    coverage = scores <= 1.
    avg_coverage = np.mean(coverage)
    normalized_coverage = np.mean(scores <= cq)

    print(f"{name} ({data}, {cal_type})",
          f"coverage: {avg_coverage * 100:.3f}%",
          f"volume: {volume:.3e}",
          f"normalized volume: {normalized_volume:.3e} ({normalized_coverage * 100:.3f}%)",
          sep=', ')
    return {
        'data': data,
        'name': name,
        'a_coverage': avg_coverage,
        'a_volume': volume,
        'n_coverage': normalized_coverage,
        'n_volume': normalized_volume,
        'cal_type': cal_type
    }

def get_score_records(df, val_set, test_set, estimator, estimator_name, n_neighbors=1., q=.9):
    X_val, diff_val = val_set
    X_test, diff_test = test_set

    df = df.append(score(X_val, diff_val, estimator, n_neighbors=n_neighbors, name=estimator_name, data='Validation', cal_type='None', q=q), ignore_index=True)
    df = df.append(score(X_test,diff_test, estimator, n_neighbors=n_neighbors, name=estimator_name, data='Test', cal_type='None', q=q), ignore_index=True)

    estimator.calibrate(X_val, diff_val, q=q, reset=True)
    df = df.append(score(X_val, diff_val, estimator, n_neighbors=n_neighbors, name=estimator_name, data='Validation', cal_type='Global', q=q), ignore_index=True)
    df = df.append(score(X_test, diff_test, estimator, n_neighbors=n_neighbors, name=estimator_name, data='Test', cal_type='Global', q=q), ignore_index=True)

    return df

def main(database_name, calibration_ratio=0.1, test_ratio=0.1, n_neighbors=1., seed=0, q=.9):
    start_time = time.time()
    reset_seed(seed)

    recovery_folder = f'./run/recovery/{database_name}/{seed}'
    if not os.path.exists(recovery_folder):
        os.makedirs(recovery_folder, exist_ok=True)

    results_folder = f'./results/sub/{database_name}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    X_train, X_test, Y_train, Y_test = get_dataset(database_name, datasets_folder='datasets', seed=seed, test_ratio=test_ratio)
    in_shape, out_shape = X_train.shape[-1], Y_train.shape[-1]
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    print(f"Dataset: {database_name}, time: {time.time() - start_time}")
    print(f"in_shape: {in_shape}, out_shape: {out_shape}")

    train_idx, cal_idx = train_test_split(np.arange(n_train), test_size=calibration_ratio, random_state=seed)
    train_r = (1-test_ratio) * (1-calibration_ratio) * 100
    cal_r = (1-test_ratio) * calibration_ratio * 100
    test_r = test_ratio * 100
    print(f"train: {len(train_idx)} ({train_r:.0f}%), calibration: {len(cal_idx)} ({cal_r:.0f}%), test: {n_test} ({test_r:.0f}%)")

    X_train, X_test = standardization_data(X_train, X_test, train_idx)

    file_url = f'{recovery_folder}/{seed}.npy'
    if os.path.isfile(file_url):
        with open(file_url, 'rb') as f:
            diff_train = np.load(f)
            diff_test = np.load(f)
    else:
        mean_model = MultiOutputRegressor(SVR())
        mean_model.fit(X_train[train_idx], Y_train[train_idx])
        diff_train = np.float32(mean_model.predict(X_train)) - Y_train
        diff_test = np.float32(mean_model.predict(X_test)) - Y_test
        with open(file_url, 'wb') as f:
            np.save(f, diff_train)
            np.save(f, diff_test)

    val_set = (X_train[cal_idx], diff_train[cal_idx])
    test_set = (X_test, diff_test)

    # load previous results
    file_url = f'{results_folder}/{seed}.q.{q}.pd.pkl'
    df = pd.DataFrame()

    """
    Estimate
    """

    # =======================
    # Baseline
    # =======================

    # Method 1: Global
    reset_seed(seed)
    estimator = BaseLearner()
    estimator.fit(X_train[train_idx], diff_train[train_idx])
    df = get_score_records(df, val_set, test_set, estimator, 'Global', n_neighbors=n_neighbors, q=q)

    # Method 2: NLE
    # based on https://copa-conference.com/papers/COPA2022_paper_7.pdf
    reset_seed(seed)
    estimator = NearestNeighborsLearner(n_neighbors=len(train_idx) // 20) # 0 = Nearest Neighbors, 1 = Global
    estimator.fit(X_train[train_idx], diff_train[train_idx])
    df = get_score_records(df, val_set, test_set, estimator, 'NLE', n_neighbors=n_neighbors, q=q)

    # prediction by NLE
    estimator.calibrate(X_train[train_idx], diff_train[train_idx], q=q, reset=True)
    pred_train = estimator(X_train[train_idx])
    lam_scores = estimator.score(X_train[train_idx], diff_train[train_idx])
    lam_volume = np.mean(np.sqrt(np.linalg.det(pred_train)))
    loss_reg = np.mean(lam_scores) / lam_volume

    print(f"baseline time: {time.time() - start_time}")

    # =======================
    # OURS
    # =======================

    experiments = []
    for lr in [1e-3, 1e-5]:
        for init_lr in [1e-3, 1e-5]:
            for dropout in [0.1, 0.5]:
                post = f'lr.{lr:.0e}.init.{init_lr:.0e}.drop.{dropout:.01f}'
                experiments.append(
                    {
                        'postfix': post, 'dropout': dropout,
                        'init_epochs': 100000, 'init_lr': init_lr, 'init_batch_size': -1,
                        'epochs': 100000, 'lr': lr, 'batch_size': -1, 'loss_reg': loss_reg,
                    }
                )

    error_occurred = False
    for e in experiments:
        reset_seed(seed)
        run_code = f'run.q.{q}'
        estimator = SGDLearner(
            optimizer_class=partial(torch.optim.Adam, lr=e['init_lr']),
            loss_func=cov_mse_loss,
            model=CovarianceModel(in_shape=in_shape, out_shape=out_shape, hidden_size=(in_shape * 4, in_shape), dropout=e['dropout'])
        )
        run_code += f'.lr.{e["init_lr"]:.0e}.epochs.{e["init_epochs"]}.batch.{e["init_batch_size"]}.dropout.{e["dropout"]}'
        pre_trained_path = f'{recovery_folder}/{run_code}.pt'

        if not os.path.isfile(pre_trained_path):
            estimator.fit(X_train[train_idx], pred_train, epochs=e['init_epochs'], batch_size=e['init_batch_size'])
            estimator.save(pre_trained_path)
        else:
            estimator.load(pre_trained_path)

        estimator.reg = 1e-3
        estimator.optimizer_class = partial(torch.optim.Adam, lr=e['lr'])
        estimator.loss_func = partial(cov_gaussian_loss, reg=e['loss_reg'])

        # load previous data
        run_code += f'.T.lr.{e["lr"]:.0e}.epochs.{e["epochs"]}.batch.{e["batch_size"]}.reg.{e["loss_reg"]:.6f}'
        trained_path = f'{recovery_folder}/{run_code}.pt'
        try:
            if not os.path.isfile(trained_path):
                estimator.fit(X_train[train_idx], diff_train[train_idx], epochs=e['epochs'], batch_size=e['batch_size'])
                estimator.save(trained_path)
            else:
                estimator.load(trained_path)
            df = get_score_records(df, val_set, test_set, estimator, f'LMVE.{e["postfix"]}', n_neighbors=n_neighbors, q=q)
        except Exception as ex:
            print(f'{e["postfix"]} failed: {ex}')
            error_occurred = True
        pd.to_pickle(df, file_url)

    if len(experiments) == 0:
        pd.to_pickle(df, file_url)

    print(f"finish time: {time.time() - start_time}")

    if error_occurred:
        raise RuntimeError

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-database', type=str, required=True)
    parser.add_argument('-seed', type=int, required=True)
    parser.add_argument('-q', type=float, default=0.9, required=False)
    args = parser.parse_args()
    main(database_name=args.database, seed=args.seed, calibration_ratio=0.1, test_ratio=0.1, n_neighbors=20, q=args.q)
