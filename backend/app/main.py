from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI(title="PhishGuard Fusion AI")


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "time": datetime.now(timezone.utc).isoformat(),
    }