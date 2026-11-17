import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from app.schemas.common import Evidence, Label, Modality, ModalityResult
from app.services.url_features import build_evidence, extract_features, normalize_url

_ROOT = Path(__file__).resolve().parents[3]  # the project root, D:\PS21
_CACHE: dict[str, tuple] = {}


def _model_dir() -> Path:
    return Path(os.environ.get("PHISHGUARD_URL_MODEL_DIR", _ROOT / "ml/artifacts/url/url_v1"))


def _load():
    d = _model_dir()
    key = str(d)
    if key in _CACHE:
        return _CACHE[key]
    if not (d / "model.joblib").exists():
        return None
    bundle = (joblib.load(d / "model.joblib"), joblib.load(d / "calibrator.joblib"),
              json.loads((d / "config.json").read_text()))
    _CACHE[key] = bundle
    return bundle


def analyze_url(raw_url: str) -> ModalityResult:
    started = time.perf_counter()
    url = normalize_url(raw_url)  # raises InvalidUrlError for bad input
    observations = build_evidence(url)
    bundle = _load()

    if bundle is None:  # no model: say so. Never pretend the URL is safe.
        return ModalityResult(
            modality=Modality.URL, available=False, evidence=observations,
            limitations=[("No trained URL model was found, so no verdict was produced. "
                         "Run the training script first.")],
        )

    model, calibrator, cfg = bundle
    host_only = cfg["feature_mode"] == "host_only"
    X = pd.DataFrame([extract_features(url, host_only=host_only)])[cfg["feature_names"]]
    p = float(calibrator.predict([float(model.predict_proba(X)[0, 1])])[0])
    label = Label.PHISHING if p >= cfg["threshold"] else Label.LEGITIMATE

    evidence = observations + [Evidence(
        id="url.model_estimate",
        description=f"URL model ({cfg['model']}) estimated a calibrated phishing probability of {p:.2f}",
        severity="info", kind="model_inference", source=cfg["version"])]

    limits = [
        "Lexical analysis only: no domain age, reputation or page content was checked.",
        "'Legitimate' means no phishing pattern was found in the text of the URL, not that the site is safe.",
    ]
    if host_only:
        limits.append("The model scores the hostname only. Path and query observations are shown "
                      "for context but were not used by the model.")

    return ModalityResult(
        modality=Modality.URL, available=True, label=label,
        probability=p, probability_calibrated=True, score=round(p * 100, 1),
        evidence=evidence, limitations=limits, model_version=cfg["version"],
        latency_ms=(time.perf_counter() - started) * 1000,
    )