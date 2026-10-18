
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_data, predict_with_confidence


app = FastAPI(
    title="Iris ML Model Serving API",
    description=(
        "Enhanced MLOps lab with model selection, "
        "confidence scores, batch predictions, "
        "and experiment metadata."
    ),
    version="1.0.0"
)

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "model" / "model_metadata.json"


# Request and Response Models

class IrisData(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


class IrisResponse(BaseModel):
    response: int


class DetailedIrisResponse(BaseModel):
    response: int
    species: str
    confidence: float


class BatchRequest(BaseModel):
    instances: list[IrisData] = Field(
        min_length=1,
        max_length=100
    )


class BatchResponse(BaseModel):
    predictions: list[DetailedIrisResponse]
    count: int


# Helper Functions

def prepare_features(iris: IrisData):
    return [
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]


def predict_safely(features, detailed=False):
    try:
        if detailed:
            return predict_with_confidence(features)
        return predict_data(features)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Run training first."
        ) from exc


# API Endpoints

@app.get("/")
def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=IrisResponse)
def predict_iris(iris_features: IrisData):
    """Original API endpoint, kept backward compatible."""
    features = [prepare_features(iris_features)]
    prediction = predict_safely(features)

    return {"response": int(prediction[0])}


@app.post(
    "/predict/detailed",
    response_model=DetailedIrisResponse
)
def predict_detailed(iris_features: IrisData):
    """Predict the flower species with class probability."""
    features = [prepare_features(iris_features)]

    result = predict_safely(features, detailed=True)
    return result[0]


@app.post(
    "/predict/batch",
    response_model=BatchResponse
)
def predict_batch(request: BatchRequest):
    """Predict multiple flower samples in one request."""
    features = [
        prepare_features(iris)
        for iris in request.instances
    ]

    results = predict_safely(features, detailed=True)

    return {
        "predictions": results,
        "count": len(results)
    }


@app.get("/model/info")
def model_info():
    """Return saved model metadata and evaluation metrics."""
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model metadata not found. Run training first."
        )

    with open(METADATA_PATH, "r") as file:
        return json.load(file)
