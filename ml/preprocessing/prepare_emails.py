import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW = Path("ml/data/raw/emails")
OUT = Path("ml/data/processed")
SOURCES = ["Enron", "Ling", "SpamAssasin", "Nigerian_Fraud", "Nazario", "CEAS_08"]
TRAIN_SOURCES = {"Enron", "Ling", "SpamAssasin", "Nigerian_Fraud"}  # held out: Nazario, CEAS_08

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ADDR_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NONWORD = re.compile(r"[^a-z0-9 ]+")


def to_text(s):
    if "<" in s and ">" in s:
        s = BeautifulSoup(s, "html.parser").get_text(" ")
    return s


def normalize(s):
    s = URL_RE.sub(" urltoken ", s)
    s = ADDR_RE.sub(" emailtoken ", s)
    s = NONWORD.sub(" ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


frames, card = [], {}
for name in SOURCES:
    df = pd.read_csv(RAW / f"{name}.csv", usecols=["subject", "body", "label"])
    n0 = len(df)
    subj = [normalize(to_text(str(x))) if pd.notna(x) else "" for x in df["subject"]]
    body = [normalize(to_text(str(x)))[:20000] if pd.notna(x) else "" for x in df["body"]]
    out = pd.DataFrame({"subject": subj, "body": body, "label": df["label"].astype(int), "source": name})
    out = out[out["body"].str.len() >= 20]
    out["text_hash"] = [hashlib.sha256((s + "|" + b).encode()).hexdigest()
                        for s, b in zip(out["subject"], out["body"])]
    out = out.drop_duplicates("text_hash")
    card[name] = {"raw": n0, "kept": len(out), "phishing_or_spam": int(out.label.sum()),
                  "legitimate": int((out.label == 0).sum())}
    frames.append(out)

alldf = pd.concat(frames, ignore_index=True).drop_duplicates("text_hash")
OUT.mkdir(parents=True, exist_ok=True)
alldf[alldf.source.isin(TRAIN_SOURCES)].to_csv(OUT / "emails_clean.csv", index=False)
alldf[~alldf.source.isin(TRAIN_SOURCES)].to_csv(OUT / "emails_independent_test.csv", index=False)
(OUT / "emails_clean.card.json").write_text(json.dumps(card, indent=2))
print(json.dumps(card, indent=2))