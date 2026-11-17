import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from app.services.url_features import extract_features
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import SEED, env_info, fit_and_evaluate, group_split


def build_X(df, host_only):
    return pd.DataFrame([extract_features(u, host_only=host_only) for u in df["url"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/processed/urls_clean.csv")
    ap.add_argument("--version", default="url_v1")
    ap.add_argument("--full-url", action="store_true", help="also use path/query (may be inflated)")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    a = ap.parse_args()
    host_only = not a.full_url

    df = pd.read_csv(a.data)
    train, val, test = group_split(df, "domain")
    print("split sizes:", len(train), len(val), len(test))
    print("largest domain groups:\n", df["domain"].value_counts().head(5))

    parts = {"tr": train, "va": val, "te": test}
    X = {k: build_X(d, host_only) for k, d in parts.items()}
    y = {k: d["label"].values for k, d in parts.items()}

    candidates = {
        "logreg": Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                       random_state=SEED))]),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=SEED, n_jobs=-1, verbose=-1),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        m, cal, thr, metrics = fit_and_evaluate(model, X["tr"], y["tr"], X["va"], y["va"],
                                                X["te"], y["te"], a.max_fpr)
        results[name], fitted[name] = metrics, (m, cal, thr)
        t = metrics["test"]
        print(f"{name}: PR-AUC={t['pr_auc']:.4f} precision={t['precision']:.4f} "
              f"recall={t['recall']:.4f} FPR={t['false_positive_rate']:.4f}")

    # Sanity check: with scrambled labels the model must do no better than chance.
    rng = np.random.default_rng(SEED)
    shuffled = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    shuffled.fit(X["tr"], rng.permutation(y["tr"]))
    sanity = float(average_precision_score(y["te"], shuffled.predict_proba(X["te"])[:, 1]))
    print(f"sanity (scrambled labels) PR-AUC={sanity:.3f} vs phishing share={y['te'].mean():.3f}")

    best = max(results, key=lambda k: results[k]["val"]["pr_auc"])  # chosen on VALIDATION only
    model, cal, thr = fitted[best]

    out = Path("ml/artifacts/url") / a.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(cal, out / "calibrator.joblib")
    (out / "config.json").write_text(json.dumps({
        "version": a.version, "model": best, "threshold": thr,
        "feature_names": list(X["tr"].columns),
        "feature_mode": "host_only" if host_only else "full_url",
        "dataset_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "env": env_info(), "max_fpr_target": a.max_fpr,
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "chosen_by": "validation PR-AUC", "chosen": best,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "sanity_scrambled_label_pr_auc": sanity, "models": results,
    }, indent=2))
    print("saved to", out, "| chosen:", best)


if __name__ == "__main__":
    main()