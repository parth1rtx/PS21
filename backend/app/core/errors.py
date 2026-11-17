import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("phishguard")


def _body(code: str, message: str, request: Request, details=None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "details": details,
        }
    }


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            _body("http_error", str(exc.detail), request), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Return only WHERE the problem is and what it is, never the submitted values.
        # A user may paste a private email into the wrong field; we must not echo it back.
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            _body("validation_error", "Invalid request", request, details), status_code=422
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # Full details go to the server log only. The user sees a generic message.
        log.exception("Unhandled error (request_id=%s)", getattr(request.state, "request_id", None))
        return JSONResponse(_body("internal_error", "Internal server error", request), status_code=500)