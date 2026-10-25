import numpy as np


class LinearRegressionScratch:
    """Ridge linear regression solved by normal equation or gradient descent.

    Parameters
    ----------
    method   : 'normal' or 'gd'
    l2       : regularization strength λ (>= 0)
    lr       : learning rate (only for 'gd')
    n_iters  : max iterations (only for 'gd')
    tol      : early-stop tolerance on |Δloss| (only for 'gd')
    """

    def __init__(self, method="normal", l2=1e-6, lr=1e-3, n_iters=10000, tol=1e-9):
        if method not in {"normal", "gd"}:
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.l2 = float(l2)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.tol = float(tol)

    # --- helpers ---
    @staticmethod
    def _add_bias(X):
        return np.column_stack([np.ones(X.shape[0]), np.asarray(X, dtype=float)])

    def _reg_matrix(self, d):
        """λ on weights only — bias (index 0) is unpenalized."""
        R = self.l2 * np.eye(d)
        R[0, 0] = 0.0
        return R

    # --- fit ---
    def fit(self, X, y):
        X = self._add_bias(X)
        y = np.asarray(y, dtype=float)
        d = X.shape[1]

        if self.method == "normal":
            # Solve (Xᵀ X + λ I) w = Xᵀ y.
            A = X.T @ X + self._reg_matrix(d)
            b = X.T @ y
            self.w_ = np.linalg.solve(A, b)
            self.loss_history_ = None
        else:  # 'gd'
            self.w_ = np.zeros(d)
            self.loss_history_ = []
            n = X.shape[0]
            prev_loss = np.inf
            for _ in range(self.n_iters):
                err = X @ self.w_ - y
                grad = (2.0 / n) * (X.T @ err) + 2.0 * self.l2 * self.w_
                # Bias excluded from penalty.
                grad[0] -= 2.0 * self.l2 * self.w_[0]
                self.w_ -= self.lr * grad
                loss = (err @ err) / n + self.l2 * np.sum(self.w_[1:] ** 2)
                self.loss_history_.append(float(loss))
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        return self

    def predict(self, X):
        X = self._add_bias(X)
        return X @ self.w_


# ---------- Metrics from scratch ----------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")