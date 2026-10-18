
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data import load_data, split_data


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"


def train_and_compare():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=12
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=12,
            n_jobs=-1
        )
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=12
    )

    comparisons = []
    best_model = None
    best_name = None
    best_f1 = -1.0

    print("=== Model Comparison ===")

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1": "f1_macro"
            }
        )

        mean_accuracy = float(scores["test_accuracy"].mean())
        mean_f1 = float(scores["test_f1"].mean())

        comparisons.append({
            "model": name,
            "cv_accuracy": round(mean_accuracy, 4),
            "cv_macro_f1": round(mean_f1, 4)
        })

        print(f"{name}: Accuracy={mean_accuracy:.4f}, F1={mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            best_name = name

    # Train the winning model
    start = time.perf_counter()
    best_model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    predictions = best_model.predict(X_test)

    test_accuracy = float(accuracy_score(y_test, predictions))
    test_f1 = float(
        f1_score(y_test, predictions, average="macro")
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    experiment = {
        "timestamp_utc": timestamp,
        "dataset": "Iris",
        "random_state": 12,
        "cv_folds": 5,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "model_comparison": comparisons,
        "selected_model": best_name,
        "training_time_seconds": round(training_time, 4),
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "macro_f1": round(test_f1, 4),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist()
        },
        "sklearn_version": sklearn.__version__
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model for FastAPI
    joblib.dump(best_model, MODEL_DIR / "iris_model.pkl")

    # Save latest experiment metrics
    with open(RESULTS_DIR / "latest_metrics.json", "w") as f:
        json.dump(experiment, f, indent=4)

    # Append to experiment history
    with open(RESULTS_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(experiment) + "\n")

    # Metadata for the model information endpoint
    metadata = {
        "model_version": "1.0.0",
        "model_name": best_name,
        "dataset": "Iris",
        "trained_at": timestamp,
        "accuracy": round(test_accuracy, 4),
        "macro_f1": round(test_f1, 4),
        "feature_names": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ],
        "class_names": [
            "setosa",
            "versicolor",
            "virginica"
        ],
        "model_path": "model/iris_model.pkl"
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n=== Final Evaluation ===")
    print(f"Selected Model: {best_name}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Training Time: {training_time:.4f}s")
    print("Experiment results and model metadata saved!")


if __name__ == "__main__":
    train_and_compare()
