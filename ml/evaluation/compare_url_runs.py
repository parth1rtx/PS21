import json
from pathlib import Path

for version in ("url_v1", "url_v1_full"):
    p = Path("ml/artifacts/url") / version / "metrics.json"
    if not p.exists():
        print(version, "-> metrics.json missing")
        continue
    m = json.loads(p.read_text())
    print(f"\n=== {version} (chosen: {m['chosen']}) ===")
    print("split sizes:", m["split_sizes"])
    print("scrambled-label PR-AUC:", round(m["sanity_scrambled_label_pr_auc"], 3))
    for name, r in m["models"].items():
        t = r["test"]
        print(f"{name:9s} PR-AUC={t['pr_auc']:.4f} precision={t['precision']:.4f} "
              f"recall={t['recall']:.4f} FPR={t['false_positive_rate']:.4f} "
              f"ECE raw={r['calibration']['ece_raw']:.4f} -> cal={r['calibration']['ece_calibrated']:.4f} "
              f"latency={r['latency_ms_per_sample']:.2f}ms")