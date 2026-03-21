from pydantic import BaseModel, Field

class SampleClip(BaseModel):
    name: str
    filename: str

class SampleResponse(BaseModel):
    clips: list[SampleClip]

class Prediction(BaseModel):
    class_: str = Field(..., alias="class")

    confidence: float

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    model_version:str = "v5"
    num_classes:int = 14

class PredictionSampleRequest(BaseModel):
    filename: str