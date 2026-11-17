import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

# Only accept safe client-supplied IDs, so nobody can inject odd characters into logs
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if SAFE_ID.match(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response