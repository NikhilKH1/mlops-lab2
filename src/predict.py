
from functools import lru_cache
from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "iris_model.pkl"
)

SPECIES_NAMES = ["setosa", "versicolor", "virginica"]


@lru_cache(maxsize=1)
def get_model():
    """Load the trained model once and cache it."""
    return joblib.load(MODEL_PATH)


def predict_data(X):
    """Original prediction interface for backward compatibility."""
    return get_model().predict(X)


def predict_with_confidence(X):
    """Return predicted species and probability for each input."""
    model = get_model()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    results = []

    for prediction, class_probs in zip(predictions, probabilities):
        class_id = int(prediction)
        class_index = list(model.classes_).index(prediction)

        results.append({
            "response": class_id,
            "species": SPECIES_NAMES[class_id],
            "confidence": round(
                float(class_probs[class_index]), 4
            )
        })

    return results
