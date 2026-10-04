"""Nested grouped cross-validation.

Outer loop: the saved fold_complex folds (no complex in both train and test).
Inner loop: GroupKFold by complex inside the outer training set; the configuration with the best
pooled Spearman over the inner out-of-fold predictions is refit on the full outer training set.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

N_INNER = 4


def complex_weights(complexes: pd.Series, scheme: str) -> np.ndarray:
    """'none' -> 1 per row; 'sqrt' -> 1/sqrt(rows of that complex in this training set)."""
    if scheme == "none":
        return np.ones(len(complexes))
    return 1.0 / np.sqrt(complexes.map(complexes.value_counts()).values)


def _fit_predict(make, params, X, y, groups, w_scheme, tr, te):
    m = make(**params)
    m.fit(X.iloc[tr], y[tr], sample_weight=complex_weights(groups.iloc[tr], w_scheme))
    return m.predict(X.iloc[te])


def nested_cv(make, grid: dict, X: pd.DataFrame, y, groups: pd.Series, outer: np.ndarray, w_scheme: str = "none",
              train_mask=None):
    """Return out-of-fold predictions for every row and the chosen params per outer fold.

    train_mask (optional boolean array): only these rows may be used for training and tuning; every row is still predicted
    when its fold is held out (used to train on single-point rows only and still score multi-point rows)."""
    y = np.asarray(y, float)
    pred, chosen = np.full(len(y), np.nan), {}
    configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    for k in np.unique(outer):
        ok = np.ones(len(y), bool) if train_mask is None else np.asarray(train_mask)
        tr, te = np.where((outer != k) & ok)[0], np.where(outer == k)[0]
        best, best_score = configs[0], -np.inf
        if len(configs) > 1:
            inner = list(GroupKFold(N_INNER).split(tr, groups=groups.iloc[tr]))
            for params in configs:
                oof = np.empty(len(tr))
                for itr, ite in inner:
                    oof[ite] = _fit_predict(make, params, X, y, groups, w_scheme, tr[itr], tr[ite])
                score = spearmanr(oof, y[tr])[0]
                if np.isfinite(score) and score > best_score:
                    best, best_score = params, score
        pred[te] = _fit_predict(make, best, X, y, groups, w_scheme, tr, te)
        chosen[int(k)] = best
    return pred, chosen
