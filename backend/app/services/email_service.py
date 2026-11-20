import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from app.schemas.common import Evidence, Label, Modality, ModalityResult
from app.services.email_features import (
    STRUCT_COLS,
    find_phrases,
    model_text,
    structural_features,
)

_ROOT = Path(__file__).resolve().parents[3]
_CACHE: dict[str, tuple] = {}


def _model_dir() -> Path:
    return Path(os.environ.get("PHISHGUARD_EMAIL_MODEL_DIR",
                               _ROOT / "ml/artifacts/email/email_v1_clean"))


def _load():
    d = _model_dir()
    if str(d) in _CACHE:
        return _CACHE[str(d)]
    if not (d / "model.joblib").exists():
        return None
    bundle = (joblib.load(d / "model.joblib"), joblib.load(d / "calibrator.joblib"),
              json.loads((d / "config.json").read_text()))
    _CACHE[str(d)] = bundle
    return bundle


def _top_terms(model, frame, k=5):
    """Words in THIS email that pushed the score toward phishing (model-derived)."""
    vec = model.named_steps["pre"].named_transformers_["tfidf"]
    terms = vec.get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0][: len(terms)]
    row = vec.transform(frame["text"]).tocoo()
    contrib = sorted(((coefs[c] * v, terms[c]) for c, v in zip(row.col, row.data)), reverse=True)
    return [t for score, t in contrib[:k] if score > 0]


def analyze_email(subject: str, body: str) -> ModalityResult:
    started = time.perf_counter()
    evidence: list[Evidence] = []
    descriptions = {
        "urgency": "urgency wording",
        "credential_request": "wording that asks for account or credential action",
        "impersonation": "generic impersonation wording",
    }
    for cat, found in find_phrases(subject, body).items():
        if found:
            evidence.append(Evidence(
                id=f"email.{cat}", severity="low", kind="observation", source="phrase_rules",
                description=f"Contains {descriptions[cat]}. Keyword matches are hints, not proof.",
                observed_value=", ".join(found[:5])))

    bundle = _load()
    if bundle is None:  # no model: say so. Never pretend the email is safe.
        return ModalityResult(
            modality=Modality.EMAIL, available=False, evidence=evidence,
            limitations=["No trained email model was found, so no verdict was produced."])

    model, cal, cfg = bundle
    frame = pd.DataFrame([{"text": model_text(subject, body),
                           **structural_features(subject, body)}])
    frame = frame[["text", *STRUCT_COLS]]  # the pipeline selects what it needs
    p = float(cal.predict([float(model.predict_proba(frame)[0, 1])])[0])
    label = Label.PHISHING if p >= cfg["threshold"] else Label.LEGITIMATE

    terms = _top_terms(model, frame)
    if terms:
        evidence.append(Evidence(
            id="email.model_terms", severity="info", kind="model_inference", source=cfg["version"],
            description="Words in this email that most increased the model's phishing estimate",
            observed_value=", ".join(terms)))
    evidence.append(Evidence(
        id="email.model_estimate", severity="info", kind="model_inference", source=cfg["version"],
        description=f"Email model estimated a calibrated phishing probability of {p:.2f}"))

    return ModalityResult(
        modality=Modality.EMAIL, available=True, label=label,
        probability=p, probability_calibrated=True, score=round(p * 100, 1),
        evidence=evidence,
        limitations=[
            ("Text only: sender authentication (SPF/DKIM), headers, attachments and link "
            "destinations were not analysed."),
            "Label 'phishing' means spam-or-phishing: training data mixes spam and phishing corpora.",
            ("Trained on public corpora whose era and style differ from modern email. On an "
            "independent source the false-positive rate was higher than the target; see the model card."),
            "A 'legitimate' label does not mean the email is safe."],
        model_version=cfg["version"], latency_ms=(time.perf_counter() - started) * 1000)