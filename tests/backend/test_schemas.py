import pytest
from app.schemas.common import Label, Modality, ModalityResult
from pydantic import ValidationError


def test_unavailable_cannot_have_verdict():
    with pytest.raises(ValidationError):
        ModalityResult(modality=Modality.URL, available=False, label=Label.LEGITIMATE)


def test_probability_needs_calibration_flag():
    with pytest.raises(ValidationError):
        ModalityResult(
            modality=Modality.URL, available=True,
            label=Label.PHISHING, probability=0.9,
        )
