"""Pydantic request/response models."""

from pydantic import BaseModel, Field, field_validator

from . import config


class LoadRequest(BaseModel):
    url: str = Field(min_length=8)


class ExtractRequest(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float
    overwrite: bool = False

    @field_validator("end_s")
    @classmethod
    def check_range(cls, end_s, info):
        start_s = info.data.get("start_s")
        if start_s is None:
            return end_s
        dur = end_s - start_s
        if dur < config.MIN_CLIP_S:
            raise ValueError(f"Clip must be at least {config.MIN_CLIP_S}s (got {dur:.2f}s)")
        if dur > config.MAX_CLIP_S:
            raise ValueError(f"Clip must be at most {config.MAX_CLIP_S}s (got {dur:.2f}s)")
        return end_s


class AnnotationIn(BaseModel):
    emotion: str
    valence: int = Field(ge=config.VALENCE_SCALE["min"], le=config.VALENCE_SCALE["max"])
    arousal: int = Field(ge=config.AROUSAL_SCALE["min"], le=config.AROUSAL_SCALE["max"])
    gender: str
    transcript: str = ""
    actor: str = ""
    notes: str = ""
    annotator_id: str = Field(min_length=1)

    @field_validator("emotion")
    @classmethod
    def check_emotion(cls, v):
        if v not in config.EMOTIONS:
            raise ValueError(f"emotion must be one of {config.EMOTIONS}")
        return v

    @field_validator("gender")
    @classmethod
    def check_gender(cls, v):
        if v not in config.GENDER_OPTIONS:
            raise ValueError(f"gender must be one of {config.GENDER_OPTIONS}")
        return v
