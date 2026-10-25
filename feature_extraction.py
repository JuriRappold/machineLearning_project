import numpy as np
import pandas as pd


# ---------- Step 5: OneHotEncoderScratch ----------
class OneHotEncoderScratch:
    """Fit on training categories, transform any set; unknown -> zero row."""

    def fit(self, values):
        self.categories_ = np.unique(np.asarray(values))
        return self

    def transform(self, values):
        values = np.asarray(values)
        # (n, 1) == (1, k) -> (n, k) indicator matrix
        return (values[:, None] == self.categories_[None, :]).astype(float)

    def fit_transform(self, values):
        return self.fit(values).transform(values)


# ---------- Step 6: StandardScalerScratch ----------
class StandardScalerScratch:
    """Z-score scaler; std == 0 -> divide by 1 (no NaN)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero on constant columns.
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------- Steps 1-4: build the raw numeric matrix + target ----------
def _build_base(df):
    """Parse datetime, sort, add calendar / cyclical / trend features.

    Returns
    -------
    X_numeric : (n, k) float array of scaled numeric features
    companies : (n,) array of company strings
    y         : (n,) float array of MW target
    """
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    ts = df["Datetime"]
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    doy = ts.dt.dayofyear.to_numpy()
    is_weekend = (dow >= 5).astype(float)

    def cyc(x, period):
        return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)

    hour_sin, hour_cos = cyc(hour, 24)
    dow_sin, dow_cos = cyc(dow, 7)
    doy_sin, doy_cos = cyc(doy, 365.25)

    years_since_start = ((ts - ts.min()).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600))

    numeric = np.column_stack([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        doy_sin, doy_cos,
        is_weekend,
        years_since_start,
    ])
    companies = df["Company"].to_numpy()
    y = df["MW"].to_numpy(dtype=float)
    return numeric, companies, y


# ---------- Step 7+9: assemble full feature matrix ----------
def build_features(df, scaler=None, encoder=None, fit=False):
    """Build the full feature matrix for a DataFrame.

    Parameters
    ----------
    df       : DataFrame with columns date, time, company, MW
    scaler   : StandardScalerScratch (or None to create a new one)
    encoder  : OneHotEncoderScratch (or None to create a new one)
    fit      : if True, fit scaler + encoder on the *training* split

    Returns
    -------
    X            : (n, k) float feature matrix
    y            : (n,) target array
    encoder      : the fitted encoder
    scaler       : the fitted scaler
    """
    numeric, companies, y = _build_base(df)

    if encoder is None:
        encoder = OneHotEncoderScratch()
    if scaler is None:
        scaler = StandardScalerScratch()

    if fit:
        scaler.fit(numeric)
        encoder.fit(companies)

    X_num = scaler.transform(numeric)
    X_cat = encoder.transform(companies)
    X = np.column_stack([X_num, X_cat])
    return X, y, encoder, scaler


def build_with_interactions(df, scaler=None, encoder=None, fit=False):
    """Like build_features, but also adds company x numeric interactions."""
    X, y, encoder, scaler = build_features(df, scaler=scaler, encoder=encoder, fit=fit)

    n_numeric = scaler.mean_.shape[0]
    n_companies = encoder.categories_.shape[0]

    X_num = X[:, :n_numeric]
    X_cat = X[:, n_numeric:]

    # Tile numeric features once per company, multiply by each company dummy column.
    interactions = np.einsum("ij,ik->ijk", X_num, X_cat).reshape(X.shape[0], n_numeric * n_companies)
    X_full = np.column_stack([X, interactions])
    return X_full, y, encoder, scaler


# ---------- Step 9: chronological split ----------
def chronological_split(df, test_frac=0.2):
    """Last `test_frac` of rows go to test; no shuffling."""
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("_ts").drop(columns="_ts").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def test():
    df = pd.read_csv("../data/all_data_READY.csv")
    DATETIME = "Datetime"
    COMPANY = "Company"
    MEGAWATTS = "MW"
    ORIGINAL_COLUMNS = [COMPANY, DATETIME, MEGAWATTS]
    print(len(df))
    print(df[COMPANY].unique())
    df[DATETIME] = pd.to_datetime(df[DATETIME], errors="raise")
    encoder = OneHotEncoderScratch()
    encoded = encoder.fit_transform(df[COMPANY])
    print(encoded)

    sets: set = set()
    for cat in encoded:
        sets.add(cat[0])
    print(sets)

#test()