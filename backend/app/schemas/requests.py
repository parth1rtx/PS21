from pydantic import BaseModel, Field, field_validator, model_validator


class UrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def strip_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL is empty")
        return v


class EmailRequest(BaseModel):
    subject: str = Field(default="", max_length=998)
    body: str = Field(default="", max_length=200_000)

    @model_validator(mode="after")
    def need_some_text(self):
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("Provide a subject or a body")
        return self