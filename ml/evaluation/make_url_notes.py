import json
from pathlib import Path

import pandas as pd

from ml.common import group_split

A = Path("ml/artifacts/url")


def load(v):
    return json.loads((A / v / "metrics.json").read_text())


m, mf = load("url_v1"), load("url_v1_full")
cfg = json.loads((A / "url_v1" / "config.json").read_text())
card = json.loads(Path("ml/data/processed/urls_clean.card.json").read_text())
df = pd.read_csv("ml/data/processed/urls_clean.csv")
tr, va, te = group_split(df, "domain")
top = df["domain"].value_counts().head(3)
chosen, chosen_f = m["chosen"], mf["chosen"]

rows = []
for name, r in m["models"].items():
    t = r["test"]
    rows.append(f"| {name} | {r['val']['pr_auc']:.4f} | {t['pr_auc']:.4f} | {t['precision']:.4f} "
                f"| {t['recall']:.4f} | {t['false_positive_rate']:.4f} |")
c = m["models"][chosen]["calibration"]
t = m["models"][chosen]["test"]
gap = mf["models"][chosen_f]["test"]["pr_auc"] - t["pr_auc"]

text = f"""# URL model notes (url_v1)

## Data
- Dataset: PhiUSIIL Phishing URL Dataset (see datasets.md)
- Rows after cleaning: {card['rows']} ({card['phishing']} phishing, {card['legitimate']} legitimate)
- Split: by registered domain, seed 42
- Split sizes: train {m['split_sizes']['train']}, val {m['split_sizes']['val']}, test {m['split_sizes']['test']}
- Features: hostname only

## Models compared (winner chosen on validation PR-AUC only)
| Model | Val PR-AUC | Test PR-AUC | Test precision | Test recall | Test FPR |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

- Chosen: {chosen}. The test set was scored once, after all choices were made.

## Calibration
- Method: isotonic regression fitted on validation
- ECE before {c['ece_raw']:.4f}, after {c['ece_calibrated']:.4f}
- Threshold {cfg['threshold']:.4f}, targeting about {cfg['max_fpr_target']*100:.0f}% false positives on validation
- At this threshold test recall is {t['recall']:.4f}

## Leakage checks
1. Scrambled labels: PR-AUC {m['sanity_scrambled_label_pr_auc']:.3f} vs test phishing share {te.label.mean():.3f}.
   Result: [WRITE: pass if close to the share]
2. Hostname-only test PR-AUC {t['pr_auc']:.4f} vs full-URL ({chosen_f}) {mf['models'][chosen_f]['test']['pr_auc']:.4f}; gap {gap:+.4f}.
   Interpretation: [WRITE: small gap = paths add little; large gap = path-style bias]
3. Largest domains: {', '.join(f'{d} ({n})' for d, n in top.items())}; unique domains {df.domain.nunique()} of {len(df)} rows.
   Phishing share: train {tr.label.mean():.3f}, val {va.label.mean():.3f}, test {te.label.mean():.3f}.
   Comment: [WRITE: balanced or lopsided]
4. Hostname-only PR-AUC is {t['pr_auc']:.4f}. Comment: [WRITE: plausible or suspiciously high]

## Limitations
- Lexical features only: no domain age, reputation, page content or WHOIS data
- Class balance is artificial; real traffic is mostly legitimate, so real precision would be lower
- Legitimate and phishing URLs in PhiUSIIL may differ in style, so some signal may be dataset-specific
- Phishing tactics change over time; collection period: [WRITE from the paper]
- "Legitimate" means no phishing pattern in the URL text, not that the site is safe

## Conclusion
[WRITE: one honest paragraph]
"""
Path("docs/url_model_notes.md").write_text(text, encoding="utf-8")
print(text)