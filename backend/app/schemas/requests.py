from pydantic import BaseModel, Field, field_validator


class UrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def strip_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL is empty")
        return v