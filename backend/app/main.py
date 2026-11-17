import logging
from datetime import UTC, datetime

from fastapi import FastAPI

from app.api import analyze
from app.core.errors import register_error_handlers
from app.core.middleware import RequestIDMiddleware

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="PhishGuard Fusion AI")
app.include_router(analyze.router, prefix="/api/v1")
app.add_middleware(RequestIDMiddleware)
register_error_handlers(app)


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "time": datetime.now(UTC).isoformat(),
    }   