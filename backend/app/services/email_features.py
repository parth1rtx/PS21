import math
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
ADDR_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NONWORD = re.compile(r"[^a-z0-9 ]+")

ARTIFACT_WORDS = {"enron", "vince", "louise", "houston", "713", "gas", "energy",
                  "linguistics", "university", "edu", "language",
                  "2000", "2001", "2002", "2004", "2005", "01"}

PHRASES = {
    "urgency": ["urgent", "immediately", "within 24 hours", "act now", "final notice",
                "expires", "suspended", "as soon as possible", "limited time"],
    "credential_request": ["verify your account", "confirm your password", "update your payment",
                           "enter your password", "log in", "login", "sign in", "security code",
                           "social security", "credit card", "bank account"],
    "impersonation": ["dear customer", "dear user", "account team", "security team",
                      "support team", "helpdesk", "it department"],
}
STRUCT_COLS = ["num_urls", "log_body_len", "subject_len",
               "urgency_hits", "credential_hits", "impersonation_hits"]


def normalize(text: str) -> str:
    text = URL_RE.sub(" urltoken ", text)
    text = ADDR_RE.sub(" emailtoken ", text)
    text = NONWORD.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def find_phrases(subject: str, body: str) -> dict[str, list[str]]:
    low = normalize(subject + " " + body)
    return {cat: [p for p in plist if p in low] for cat, plist in PHRASES.items()}


def structural_features(subject: str, body: str) -> dict[str, float]:
    s, b = normalize(subject), normalize(body)
    hits = find_phrases(s, b)
    return {
        "num_urls": (s + " " + b).count("urltoken"),
        "log_body_len": math.log1p(len(b)),
        "subject_len": len(s),
        "urgency_hits": len(hits["urgency"]),
        "credential_hits": len(hits["credential_request"]),
        "impersonation_hits": len(hits["impersonation"]),
    }


def model_text(subject: str, body: str) -> str:
    words = normalize(subject + " " + body).split()
    return " ".join(w for w in words if w not in ARTIFACT_WORDS)