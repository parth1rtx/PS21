import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from app.services.url_features import InvalidUrlError, normalize_url, registered_domain

SRC = Path("ml/data/raw/phiusiil.csv")
OUT = Path("ml/data/processed/urls_clean.csv")
PHISHING_IS_ZERO = True  # confirm on the dataset page, then keep or change

df = pd.read_csv(SRC, usecols=["URL", "label"])
rows, dropped = [], 0
for raw, lab in zip(df["URL"].astype(str), df["label"]):
    try:
        url = normalize_url(raw)
    except InvalidUrlError:
        dropped += 1
        continue
    host = (urlsplit(url).hostname or "").lower()
    if "." not in host:
        dropped += 1
        continue
    is_phish = int(lab == 0) if PHISHING_IS_ZERO else int(lab == 1)
    rows.append({"url": url, "host": host, "domain": registered_domain(host),
                 "label": is_phish, "source": "phiusiil"})

out = pd.DataFrame(rows).drop_duplicates("url")
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

card = {"rows": len(out), "phishing": int(out.label.sum()),
        "legitimate": int((out.label == 0).sum()), "dropped_invalid": dropped,
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest()}
OUT.with_suffix(".card.json").write_text(json.dumps(card, indent=2))
print(card)