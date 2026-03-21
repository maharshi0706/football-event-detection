import sys
sys.path.append("..")
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import  File, FastAPI, UploadFile, HTTPException

from Api.schemas import SampleClip, SampleResponse, PredictionResponse, PredictionSampleRequest, Prediction
from Inference.modelTesting import load_model, predict

MODEL_VERSION = "v5"
NUM_CLASSES = 14

SAMPLE_FOLDER = Path(r"D:\Football Event Detection\Samples")
WEIGHTS_PATH = Path(r"D:\Football Event Detection\ML\checkpoints\best_acc_0.6627.pth")

ALLOWED_TYPES = {".mp4", ".avi", ".mov"}

_model = None
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(
    title="Football Event Detection API",
    description="VideoMAE fine-tuned on SoccerNet - 14 football event classes",
    version=MODEL_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501/"],
    allow_methods=["get","post"],
    allow_headers=["*"]
)


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            print(f"Loading model from {WEIGHTS_PATH}")
            _model = load_model(str(WEIGHTS_PATH))
            print("Model Loaded.")

    return _model


def _build_predict_raw(raw_predictions: list[dict]) -> PredictionResponse:
    predictions = [
        Prediction(**{"class": p["class"], "confidence": p["confidence"]}) 
        for p in raw_predictions
    ]
    
    return PredictionResponse(
        predictions=predictions,
        model_version=MODEL_VERSION,
        num_classes=NUM_CLASSES
    )

@app.get("/")
async def root():
    return {"message": "Hello world"}


@app.get("/samples", response_model=SampleResponse)
async def getSampleVideoNames():
    if not SAMPLE_FOLDER.exists():
        return SampleResponse(clips=[])
    
    sampleClips = []
    for f in sorted(SAMPLE_FOLDER.iterdir()):
        if f.suffix in ALLOWED_TYPES:
            name = f.stem.replace("_"," ").title()
            sampleClips.append(SampleClip(name=name, filename=f.name))

    # logging.log(sampleClips)
    return SampleResponse(clips=sampleClips)


@app.get("/sample-video/{filename}")
async def getSampleVideo(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail=f"Invalid File Name")
    
    clip_path = SAMPLE_FOLDER / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample video not found:{filename}")
    
    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename
    )


@app.post("/predict-sample", response_model=PredictionResponse)
async def predictSample(body: PredictionSampleRequest):
    if "/" in body.filename or "\\" in body.filename or ".." in body.filename:
        raise HTTPException(status_code=400, detail=f"Invalid Filename: {body.filename}")
    
    clip_path = SAMPLE_FOLDER / body.filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="File does not exist.")
    
    try:
        model = get_model()
        raw = predict(model, str(clip_path), top_k=3)
        return _build_predict_raw(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failed: {e}")
    