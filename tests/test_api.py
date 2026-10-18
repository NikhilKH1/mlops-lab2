
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow tests to import the existing src/main.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main
import predict


client = TestClient(main.app)

VALID_FLOWER = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

SECOND_FLOWER = {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3
}


def test_health_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_single_prediction():
    response = client.post(
        "/predict", json=VALID_FLOWER
    )

    assert response.status_code == 200
    assert response.json() == {"response": 0}


def test_detailed_prediction():
    response = client.post(
        "/predict/detailed", json=VALID_FLOWER
    )

    assert response.status_code == 200

    data = response.json()

    assert data["response"] == 0
    assert data["species"] == "setosa"
    assert 0 <= data["confidence"] <= 1


def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={
            "instances": [
                VALID_FLOWER,
                SECOND_FLOWER
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["species"] == "setosa"


@pytest.mark.parametrize(
    "field",
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_measurements(field, value):
    invalid_flower = VALID_FLOWER.copy()
    invalid_flower[field] = value

    response = client.post(
        "/predict", json=invalid_flower
    )

    assert response.status_code == 422


def test_missing_required_field():
    invalid_flower = VALID_FLOWER.copy()
    del invalid_flower["sepal_length"]

    response = client.post(
        "/predict", json=invalid_flower
    )

    assert response.status_code == 422


def test_empty_batch():
    response = client.post(
        "/predict/batch",
        json={"instances": []}
    )

    assert response.status_code == 422


def test_batch_limit():
    response = client.post(
        "/predict/batch",
        json={
            "instances": [VALID_FLOWER] * 101
        }
    )

    assert response.status_code == 422


def test_model_metadata():
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "1.0.0"
    assert data["model_name"] in [
        "Decision Tree",
        "Random Forest"
    ]
    assert 0 <= data["accuracy"] <= 1


def test_missing_metadata(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing.json"
    monkeypatch.setattr(
        main, "METADATA_PATH", missing_file
    )

    response = client.get("/model/info")

    assert response.status_code == 503


def test_model_unavailable(monkeypatch):
    def unavailable_model():
        raise FileNotFoundError("Model unavailable")

    monkeypatch.setattr(
        predict, "get_model", unavailable_model
    )

    response = client.post(
        "/predict", json=VALID_FLOWER
    )

    assert response.status_code == 503
