import itertools
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def group_split(df, group_col, seed=SEED, val=0.15, test=0.15):
    """Split so no group (website domain) appears in two piles."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    tv_idx, te_idx = next(g1.split(df, groups=df[group_col]))
    tv = df.iloc[tv_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    tr_idx, va_idx = next(g2.split(tv, groups=tv[group_col]))
    return tv.iloc[tr_idx], tv.iloc[va_idx], df.iloc[te_idx]


def threshold_for_fpr(y, p, max_fpr=0.01):
    """Pick the threshold so about max_fpr of safe items get wrongly flagged."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return 0.5
    return float(min(1.0, np.nextafter(np.quantile(neg, 1 - max_fpr), 1)))


def classification_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": len(y), "n_positive": int(np.sum(y)),
    }


def ece(y, p, bins=10):
    """Expected calibration error: average gap between predicted and actual phishing rate."""
    edges = np.linspace(0, 1, bins + 1)
    total, err = len(y), 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & ((p < hi) | (hi == 1.0))
        if m.any():
            err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return float(err)


def env_info():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED}
    for pkg in ["scikit-learn", "pandas", "numpy", "lightgbm", "joblib", "tldextract"]:
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = "unknown"
    return out


def fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, X_te, y_te, max_fpr=0.01):
    """Train on train; calibrate and choose the threshold on validation; score test once."""
    model.fit(X_tr, y_tr)
    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
    c_val, c_te = cal.predict(p_val), cal.predict(p_te)
    thr = threshold_for_fpr(y_val, c_val, max_fpr)

    n = min(200, len(X_te))
    t0 = time.perf_counter()
    for i in range(n):
        model.predict_proba(X_te.iloc[[i]])
    latency = (time.perf_counter() - t0) / n * 1000

    metrics = {
        "val": classification_metrics(y_val, c_val, thr),
        "test": classification_metrics(y_te, c_te, thr),
        "test_pr_auc_raw": float(average_precision_score(y_te, p_te)),
        "calibration": {
            "ece_raw": ece(y_te, p_te), "ece_calibrated": ece(y_te, c_te),
            "brier_raw": float(brier_score_loss(y_te, p_te)),
            "brier_calibrated": float(brier_score_loss(y_te, c_te)),
        },
        "latency_ms_per_sample": latency,
    }
    return model, cal, thr, metrics