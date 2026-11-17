from fastapi import APIRouter, HTTPException

from app.schemas.common import ModalityResult
from app.schemas.requests import UrlRequest
from app.services.url_features import InvalidUrlError
from app.services.url_service import analyze_url

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/url", response_model=ModalityResult)
def analyze_url_endpoint(req: UrlRequest):
    try:
        return analyze_url(req.url)
    except InvalidUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc