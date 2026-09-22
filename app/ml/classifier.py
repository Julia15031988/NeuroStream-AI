from pathlib import Path

import joblib

from app.ml.features import build_feature_vector

MODEL_PATH = Path(__file__).resolve().parent / "models" / "state_classifier.joblib"

model = joblib.load(MODEL_PATH)


def classify_state(
    delta_power: float,
    theta_power: float,
    alpha_power: float,
    beta_power: float,
    attention_score: float,
    relaxation_score: float,
) -> dict:
    feature_vector = build_feature_vector(
        delta_power=delta_power,
        theta_power=theta_power,
        alpha_power=alpha_power,
        beta_power=beta_power,
        attention_score=attention_score,
        relaxation_score=relaxation_score,
    )

    prediction = model.predict([feature_vector])

    probabilities = model.predict_proba([feature_vector])

    predicted_state = str(prediction[0])

    confidence = float(max(probabilities[0]))

    return {
        "state": predicted_state,
        "confidence": round(
            confidence,
            4,
        ),
    }
