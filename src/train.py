
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score

from data import load_data, split_data


def train_and_compare():
    # Load and split the Iris dataset
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Define candidate models
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

    # Reproducible 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=12
    )

    best_model = None
    best_name = None
    best_f1 = -1.0

    print("=== Model Comparison ===")

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1": "f1_macro"
            }
        )

        mean_accuracy = scores["test_accuracy"].mean()
        mean_f1 = scores["test_f1"].mean()

        print(f"\nModel: {name}")
        print(f"CV Accuracy: {mean_accuracy:.4f}")
        print(f"CV Macro F1: {mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            best_name = name

    # Fit selected model using all training data
    best_model.fit(X_train, y_train)

    # Final evaluation on untouched test data
    predictions = best_model.predict(X_test)

    print("\n=== Selected Model ===")
    print(f"Model: {best_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Test Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")

    # Preserve the existing FastAPI model path
    model_path = (
        Path(__file__).resolve().parent.parent
        / "model"
        / "iris_model.pkl"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_and_compare()
