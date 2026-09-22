from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.ml.features import build_feature_vector
from app.signal_processing.analysis import (
    calculate_attention_score,
    calculate_relaxation_score,
)
from app.signal_processing.features import (
    calculate_band_power,
    calculate_psd,
)
from app.signal_processing.preprocessing import (
    bandpass_filter,
    remove_dc_offset,
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "state_classifier.joblib"
)

SAMPLING_RATE = 256
SAMPLES_COUNT = 256


def generate_signal(
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    theta: float,
    delta: float,
    noise: float,
) -> list[float]:
    time = (
        np.arange(SAMPLES_COUNT)
        / SAMPLING_RATE
    )

    signal = (
        alpha
        * np.sin(
            2 * np.pi * 10 * time
        )
        + beta
        * np.sin(
            2 * np.pi * 20 * time
        )
        + theta
        * np.sin(
            2 * np.pi * 6 * time
        )
        + delta
        * np.sin(
            2 * np.pi * 2 * time
        )
        + rng.normal(
            0,
            noise,
            SAMPLES_COUNT,
        )
    )

    return signal.tolist()


def extract_features(
    samples: list[float],
) -> list[float]:
    cleaned_samples = remove_dc_offset(
        samples,
    )

    filtered_samples = bandpass_filter(
        cleaned_samples,
        SAMPLING_RATE,
    )

    frequencies, power_spectral_density = (
        calculate_psd(
            filtered_samples,
            SAMPLING_RATE,
        )
    )

    delta_power = calculate_band_power(
        frequencies,
        power_spectral_density,
        0.5,
        4.0,
    )

    theta_power = calculate_band_power(
        frequencies,
        power_spectral_density,
        4.0,
        8.0,
    )

    alpha_power = calculate_band_power(
        frequencies,
        power_spectral_density,
        8.0,
        13.0,
    )

    beta_power = calculate_band_power(
        frequencies,
        power_spectral_density,
        13.0,
        30.0,
    )

    attention_score = (
        calculate_attention_score(
            theta_power,
            beta_power,
        )
    )

    relaxation_score = (
        calculate_relaxation_score(
            alpha_power,
            beta_power,
        )
    )

    return build_feature_vector(
        delta_power=delta_power,
        theta_power=theta_power,
        alpha_power=alpha_power,
        beta_power=beta_power,
        attention_score=attention_score,
        relaxation_score=relaxation_score,
    )


def generate_dataset(
    samples_per_state: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    features = []
    labels = []

    for _ in range(samples_per_state):
        relaxed_signal = generate_signal(
            rng=rng,
            alpha=rng.uniform(0.55, 1.0),
            beta=rng.uniform(0.20, 0.65),
            theta=rng.uniform(0.20, 0.65),
            delta=rng.uniform(0.10, 0.40),
            noise=rng.uniform(0.20, 0.60),
        )

        features.append(
            extract_features(
                relaxed_signal,
            )
        )
        labels.append("relaxed")

        focused_signal = generate_signal(
            rng=rng,
            alpha=rng.uniform(0.20, 0.65),
            beta=rng.uniform(0.55, 1.0),
            theta=rng.uniform(0.20, 0.65),
            delta=rng.uniform(0.10, 0.35),
            noise=rng.uniform(0.20, 0.60),
        )

        features.append(
            extract_features(
                focused_signal,
            )
        )
        labels.append("focused")

        drowsy_signal = generate_signal(
            rng=rng,
            alpha=rng.uniform(0.20, 0.65),
            beta=rng.uniform(0.10, 0.45),
            theta=rng.uniform(0.55, 1.0),
            delta=rng.uniform(0.25, 0.70),
            noise=rng.uniform(0.20, 0.60),
        )

        features.append(
            extract_features(
                drowsy_signal,
            )
        )
        labels.append("drowsy")

    return (
        np.asarray(
            features,
            dtype=float,
        ),
        np.asarray(labels),
    )


def train_model() -> None:
    x, y = generate_dataset()

    x_train, x_test, y_train, y_test = (
        train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
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

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train_model()
