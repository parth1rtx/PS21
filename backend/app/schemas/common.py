from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Modality(str, Enum):
    URL = "url"
    EMAIL = "email"
    SCREENSHOT = "screenshot"
    CREDENTIAL = "credential"


class Label(str, Enum):
    PHISHING = "phishing"
    LEGITIMATE = "legitimate"
    UNKNOWN = "unknown"  # not enough evidence; NEVER treated as safe


class Evidence(BaseModel):
    id: str                          # e.g. "url.ip_as_host"
    description: str                 # what was actually observed
    severity: str = "info"           # info / low / medium / high
    observed_value: str | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModalityResult(BaseModel):
    modality: Modality
    available: bool
    label: Label = Label.UNKNOWN
    probability: float | None = Field(default=None, ge=0, le=1)
    probability_calibrated: bool = False
    score: float | None = Field(default=None, ge=0, le=100)
    evidence: list[Evidence] = []
    limitations: list[str] = []
    model_version: str | None = None

    @model_validator(mode="after")
    def check_consistency(self):
        if not self.available and (
            self.label != Label.UNKNOWN
            or self.probability is not None
            or self.score is not None
        ):
            raise ValueError("An unavailable modality cannot carry a verdict")
        if self.probability is not None and not self.probability_calibrated:
            raise ValueError("probability requires probability_calibrated=True")
        return self