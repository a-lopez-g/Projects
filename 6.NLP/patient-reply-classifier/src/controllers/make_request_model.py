from typing import List
from pydantic import BaseModel


class RequestModel(BaseModel):
    intent: List[str]
    asr: List[str]
    lang: str

