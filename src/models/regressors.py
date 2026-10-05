"""Regressors used in the ablations. All take sample_weight and a `loss` of "huber" or "mse"."""
import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HUBER_DELTA = 1.0  # kcal/mol


def _n_components(n_pca, X):
    """Requested PCA size, capped by what the data allows (matters only for very small training sets)."""
    return min(n_pca, X.shape[0] - 1, X.shape[1])


class MeanBaseline:
    """(Weighted) mean of the training labels; `loss` is accepted for a uniform interface and ignored."""

    def __init__(self, loss="huber"):
        pass

    def fit(self, X, y, sample_weight=None):
        self.mu = np.average(y, weights=sample_weight)
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class AugmentedRidge:
    """Ridge on standardised features where the `boost_cols` (zero-shot scores) are penalised less.

    Following Hsu et al. 2022: a boosted column is multiplied by `boost` after scaling, which is equivalent
    to dividing its L2 penalty by boost**2. `scalar_cols` (e.g. geometry) are standardised and penalised
    normally but bypass the optional PCA, which only compresses the remaining (embedding) columns.
    loss="huber" minimises sum w * huber(residual, HUBER_DELTA) + alpha * ||beta||^2 with L-BFGS.

    cens_col (optional): name of a 0/1 column of X marking right-censored rows, whose y is a lower bound L on the true value
    (Tobit-style, one-sided Huber). Such a row contributes huber(max(0, L - prediction)): a prediction at or above the bound costs
    nothing. The column is not used as a feature. Only for loss="huber" (the objective stays convex).
    """

    def __init__(self, alpha=1.0, boost_cols=(), boost=10.0, n_pca=None, scalar_cols=(), loss="huber",
                 track_history=False, cens_col=None):
        self.alpha, self.boost_cols, self.boost, self.n_pca, self.loss = alpha, list(boost_cols), boost, n_pca, loss
        self.scalar_cols, self.track_history, self.cens_col = list(scalar_cols), track_history, cens_col

    def _transform(self, X, fit=False):
        if self.cens_col is not None:
            X = X.drop(columns=[self.cens_col])
        blocks = {"boost": X[self.boost_cols].values, "scalar": X[self.scalar_cols].values,
                  "emb": X.drop(columns=self.boost_cols + self.scalar_cols).values}
        if fit:  # an empty block gets no scaler
            self.scalers = {k: StandardScaler().fit(v) for k, v in blocks.items() if v.shape[1]}
        Z = {k: self.scalers[k].transform(v) if k in self.scalers else v for k, v in blocks.items()}
        if fit:
            self.pca = PCA(_n_components(self.n_pca, Z["emb"]), random_state=0).fit(Z["emb"]) if self.n_pca else None
        if self.pca is not None:
            Z["emb"] = self.pca.transform(Z["emb"]) / np.sqrt(self.pca.explained_variance_[None, :])  # unit variance
        return np.hstack([Z["boost"] * self.boost, Z["scalar"], Z["emb"]])

    def fit(self, X, y, sample_weight=None):
        cens = X[self.cens_col].values.astype(bool) if self.cens_col is not None else np.zeros(len(y), bool)
        assert self.loss == "huber" or not cens.any(), "censored rows need the Huber loss"
        Z, y = self._transform(X, fit=True), np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.mean()
        if self.loss == "mse":
            m = Ridge(alpha=self.alpha).fit(Z, y, sample_weight=w)
            self.coef_, self.intercept_ = m.coef_, m.intercept_
            return self

        def f(p):
            b, c = p[:-1], p[-1]
            r = y - Z @ b - c
            r = np.where(cens, np.maximum(r, 0.0), r)  # censored: only a prediction below the bound is penalised
            a = np.abs(r)
            quad = a <= HUBER_DELTA
            loss = np.where(quad, 0.5 * r ** 2, HUBER_DELTA * (a - 0.5 * HUBER_DELTA))
            dl = np.where(quad, r, HUBER_DELTA * np.sign(r)) * w  # weighted d loss / d residual
            return (w * loss).sum() + self.alpha * b @ b, np.append(-Z.T @ dl + 2 * self.alpha * b, -dl.sum())

        # objective after each L-BFGS iteration; costs one extra evaluation per iteration, so diagnostics only
        self.history_ = []
        record = (lambda pk: self.history_.append(f(pk)[0])) if self.track_history else None
        p0 = np.append(np.zeros(Z.shape[1]), np.median(y))
        res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 2000}, callback=record)
        self.converged_, self.n_iter_ = bool(res.success), int(res.nit)
        # at the minimum of this convex objective the gradient is 0; report its size relative to the starting point
        self.grad_norm_, self.grad_norm0_ = float(np.abs(res.jac).max()), float(np.abs(f(p0)[1]).max())
        self.coef_, self.intercept_ = res.x[:-1], res.x[-1]
        return self

    def predict(self, X):
        return self._transform(X) @ self.coef_ + self.intercept_


class PCALightGBM:
    """LightGBM where the embedding columns are first compressed with PCA (fit on the training fold);
    `keep_cols` (zero-shot scores, scalar features) bypass the PCA."""

    def __init__(self, n_pca=16, keep_cols=(), num_leaves=7, n_estimators=300, loss="huber", seed=0):
        self.n_pca, self.keep_cols, self.num_leaves, self.loss, self.seed = n_pca, list(keep_cols), num_leaves, loss, seed
        self.n_estimators = n_estimators

    def _transform(self, X, fit=False):
        Xe = X.drop(columns=self.keep_cols).values
        if not Xe.shape[1]:  # scalar features only, nothing to compress
            return X[self.keep_cols].values
        if fit:
            self.sc = StandardScaler().fit(Xe)
            self.pca = PCA(_n_components(self.n_pca, Xe), random_state=0).fit(self.sc.transform(Xe))
        return np.hstack([X[self.keep_cols].values, self.pca.transform(self.sc.transform(Xe))])

    def fit(self, X, y, sample_weight=None, eval_sets=()):
        """eval_sets: [(X, y), ...] scored every boosting round (diagnostics only); see `evals_result_`."""
        objective, metric = ("huber", "huber") if self.loss == "huber" else ("regression", "l2")
        self.m = lgb.LGBMRegressor(
            objective=objective, alpha=HUBER_DELTA,
            n_estimators=self.n_estimators, learning_rate=0.03, num_leaves=self.num_leaves, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=self.seed, n_jobs=1, verbose=-1)  # one thread: runs are parallelised across CV jobs
        Z = self._transform(X, fit=True)
        evals = [(self._transform(Xe), ye) for Xe, ye in eval_sets]
        self.m.fit(Z, y, sample_weight=sample_weight, eval_set=evals or None, eval_metric=metric)
        self.evals_result_ = self.m.evals_result_ if evals else {}
        return self

    def predict(self, X):
        return self.m.predict(self._transform(X))
