
import json
from pathlib import Path

import matplotlib

# Supports headless environments such as GitHub Actions.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "results" / "latest_metrics.json"
OUTPUT_DIR = ROOT / "results" / "figures"

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def plot_model_comparison(experiment):
    """Compare cross-validation scores for all models."""

    comparisons = experiment["model_comparison"]

    names = [item["model"] for item in comparisons]
    accuracies = [item["cv_accuracy"] for item in comparisons]
    f1_scores = [item["cv_macro_f1"] for item in comparisons]

    x = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    accuracy_bars = ax.bar(
        [i - width / 2 for i in x],
        accuracies,
        width,
        label="CV Accuracy"
    )

    f1_bars = ax.bar(
        [i + width / 2 for i in x],
        f1_scores,
        width,
        label="CV Macro F1"
    )

    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Cross-validation score")
    ax.set_title("Iris Classification: Model Comparison")

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Saved: {output}")


def plot_confusion_matrix(experiment):
    """Visualize held-out test predictions."""

    matrix = experiment["test_metrics"]["confusion_matrix"]
    selected_model = experiment["selected_model"]

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ticks = list(range(len(CLASS_NAMES)))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")
    ax.set_title(f"Confusion Matrix: {selected_model}")

    maximum = max(max(row) for row in matrix) or 1

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > maximum / 2
                    else "black"
                )
            )

    fig.tight_layout()

    output = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Saved: {output}")


def main():
    if not METRICS_PATH.exists():
        raise SystemExit(
            "Metrics not found. Run python src/train.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "r") as file:
        experiment = json.load(file)

    print("=== Generating MLOps Visualizations ===")

    plot_model_comparison(experiment)
    plot_confusion_matrix(experiment)

    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
