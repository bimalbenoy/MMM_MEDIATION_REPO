import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import ParameterGrid

def blocked_expanding_cv(X, y, n_folds=5):
    n = len(X)
    fold_starts = np.linspace(int(n*0.5), n-1, n_folds, dtype=int)
    for train_end in fold_starts:
        val_end = min(train_end + max(4, int(0.1*n)), n)
        yield (X.index[:train_end], X.index[train_end:val_end])

def metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
    }

def ts_grid_search(X, y, param_grid, n_folds=5, random_state=42):
    best = None
    best_rmse = np.inf
    history = []
    for params in ParameterGrid(param_grid):
        rmses = []
        for tr_idx, va_idx in blocked_expanding_cv(X, y, n_folds=n_folds):
            Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
            Xva, yva = X.loc[va_idx], y.loc[va_idx]
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('enet', ElasticNet(random_state=random_state, **params))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            rmses.append(mean_squared_error(yva, pred, squared=False))
        rm = float(np.mean(rmses))
        history.append((params, rm))
        if rm < best_rmse:
            best_rmse, best = rm, params
    return best, best_rmse, history

def fit_enet(X, y, params, random_state=42):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('enet', ElasticNet(random_state=random_state, **params))])
    pipe.fit(X, y)
    return pipe
