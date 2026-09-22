from pydantic import BaseModel, Field


class EEGPacket(BaseModel):
    channel: str = Field(min_length=1)
    sampling_rate: int = Field(gt=0)
    samples: list[float] = Field(min_length=1)
