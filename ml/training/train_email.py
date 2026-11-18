import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from app.services.email_features import STRUCT_COLS, model_text, structural_features
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import SEED, classification_metrics, env_info, fit_and_evaluate


def make_frame(df):
    subj, body = df["subject"].fillna(""), df["body"].fillna("")
    text = [model_text(s, b) for s, b in zip(subj, body)]
    struct = pd.DataFrame([structural_features(s, b) for s, b in zip(subj, body)])
    out = struct[STRUCT_COLS].copy()
    out.insert(0, "text", text)
    return out


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                           sublinear_tf=True, strip_accents="unicode")


def build(kind):
    if kind == "tfidf":
        pre = ColumnTransformer([("tfidf", tfidf(), "text")])
    else:
        pre = ColumnTransformer([("tfidf", tfidf(), "text"),
                                 ("num", StandardScaler(), STRUCT_COLS)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=SEED))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/processed/emails_clean.csv")
    ap.add_argument("--indep", default="ml/data/processed/emails_independent_test.csv")
    ap.add_argument("--version", default="email_v1")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    tv, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=tv["label"], random_state=SEED)
    parts = {"tr": tr, "va": va, "te": te}
    X = {k: make_frame(d) for k, d in parts.items()}
    y = {k: d["label"].values for k, d in parts.items()}

    results, fitted = {}, {}
    for name in ("tfidf", "tfidf_plus_structural"):
        m, cal, thr, metrics = fit_and_evaluate(build(name), X["tr"], y["tr"], X["va"], y["va"],
                                                X["te"], y["te"], a.max_fpr)
        results[name], fitted[name] = metrics, (m, cal, thr)
        t = metrics["test"]
        print(f"[in-source] {name}: PR-AUC={t['pr_auc']:.4f} precision={t['precision']:.4f} "
              f"recall={t['recall']:.4f} FPR={t['false_positive_rate']:.4f}")

    best = max(results, key=lambda k: results[k]["val"]["pr_auc"])  # chosen on validation only
    model, cal, thr = fitted[best]

    # Independent test: sources never used in training
    ind = pd.read_csv(a.indep)
    independent = {}
    for src, d in ind.groupby("source"):
        Xi = make_frame(d)
        p = cal.predict(model.predict_proba(Xi)[:, 1])
        yi = d["label"].values
        if yi.min() == yi.max():  # one class only: report recall (or specificity) only
            hit = float(((p >= thr).astype(int) == yi).mean())
            independent[src] = {"n": len(d), "only_class": int(yi[0]),
                                "fraction_correct": hit}
            print(f"[independent] {src}: n={len(d)} fraction correct={hit:.4f}")
        else:
            independent[src] = classification_metrics(yi, p, thr)
            r = independent[src]
            print(f"[independent] {src}: PR-AUC={r['pr_auc']:.4f} precision={r['precision']:.4f} "
                  f"recall={r['recall']:.4f} FPR={r['false_positive_rate']:.4f}")

    # Leak review: which words matter most?
    vec = model.named_steps["pre"].named_transformers_["tfidf"]
    terms = vec.get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0][: len(terms)]
    order = coefs.argsort()
    legit_terms, phish_terms = list(terms[order[:30]]), list(terms[order[-30:]])
    print("TOP legitimate terms:", legit_terms)
    print("TOP phishing/spam terms:", phish_terms)

    out = Path("ml/artifacts/email") / a.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(cal, out / "calibrator.joblib")
    (out / "config.json").write_text(json.dumps({
        "version": a.version, "model": best, "threshold": thr,
        "dataset_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        "env": env_info(), "max_fpr_target": a.max_fpr,
        "split": "stratified random after dedupe; independent test = held-out sources",
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "chosen_by": "validation PR-AUC", "chosen": best,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "models": results, "independent": independent,
        "top_terms": {"legitimate": legit_terms, "phishing_or_spam": phish_terms},
    }, indent=2))
    print("saved to", out, "| chosen:", best)


if __name__ == "__main__":
    main()