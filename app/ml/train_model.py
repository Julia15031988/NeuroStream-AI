from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent / "models" / "state_classifier.joblib"


def generate_dataset(
    samples_per_state: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    features = []
    labels = []

    for _ in range(samples_per_state):
        # RELAXED
        delta = rng.uniform(0.05, 0.15)
        theta = rng.uniform(0.05, 0.20)
        alpha = rng.uniform(0.50, 0.75)
        beta = rng.uniform(0.05, 0.15)

        total = delta + theta + alpha + beta

        features.append(
            [
                delta / total,
                theta / total,
                alpha / total,
                beta / total,
                rng.uniform(0.05, 0.35),
                rng.uniform(0.70, 1.00),
            ]
        )

        labels.append("relaxed")

        # FOCUSED
        delta = rng.uniform(0.05, 0.15)
        theta = rng.uniform(0.05, 0.20)
        alpha = rng.uniform(0.10, 0.30)
        beta = rng.uniform(0.40, 0.70)

        total = delta + theta + alpha + beta

        features.append(
            [
                delta / total,
                theta / total,
                alpha / total,
                beta / total,
                rng.uniform(0.65, 1.00),
                rng.uniform(0.05, 0.45),
            ]
        )

        labels.append("focused")

        # DROWSY
        delta = rng.uniform(0.10, 0.30)
        theta = rng.uniform(0.45, 0.70)
        alpha = rng.uniform(0.05, 0.20)
        beta = rng.uniform(0.03, 0.12)

        total = delta + theta + alpha + beta

        features.append(
            [
                delta / total,
                theta / total,
                alpha / total,
                beta / total,
                rng.uniform(0.05, 0.30),
                rng.uniform(0.30, 0.70),
            ]
        )

        labels.append("drowsy")

    return (
        np.asarray(features, dtype=float),
        np.asarray(labels),
    )


def train_model() -> None:
    x, y = generate_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_test,
    )

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
