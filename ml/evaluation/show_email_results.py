import json
import sys
from pathlib import Path

version = sys.argv[1] if len(sys.argv) > 1 else "email_v1"
m = json.loads((Path("ml/artifacts/email") / version / "metrics.json").read_text())
print("chosen:", m["chosen"], "| split sizes:", m["split_sizes"])
for name, r in m["models"].items():
    t = r["test"]
    print(f"[in-source] {name}: PR-AUC={t['pr_auc']:.4f} precision={t['precision']:.4f} "
          f"recall={t['recall']:.4f} FPR={t['false_positive_rate']:.4f} "
          f"ECE {r['calibration']['ece_raw']:.4f}->{r['calibration']['ece_calibrated']:.4f}")
for src, r in m["independent"].items():
    print("[independent]", src, {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in r.items() if k != "confusion_matrix"})
print("TOP legitimate:", m["top_terms"]["legitimate"])
print("TOP phishing/spam:", m["top_terms"]["phishing_or_spam"])