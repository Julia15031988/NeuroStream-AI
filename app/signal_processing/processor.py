import math
from statistics import mean

from app.ml.classifier import classify_state
from app.signal_processing.analysis import (
    calculate_attention_score,
    calculate_relaxation_score,
    detect_dominant_wave,
)
from app.signal_processing.features import (
    calculate_band_power,
    calculate_fft,
)
from app.signal_processing.preprocessing import (
    bandpass_filter,
    remove_dc_offset,
)


def process_eeg_samples(
    samples: list[float],
    sampling_rate: int = 256,
) -> dict:
    if not samples:
        return {
            "samples_count": 0,
            "average": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "std": 0.0,
            "rms": 0.0,
            "delta_power": 0.0,
            "theta_power": 0.0,
            "alpha_power": 0.0,
            "beta_power": 0.0,
            "dominant_wave": None,
            "attention_score": 0.0,
            "relaxation_score": 0.0,
            "predicted_state": None,
            "prediction_confidence": 0.0,
        }

    average = mean(samples)

    variance = sum((sample - average) ** 2 for sample in samples) / len(samples)

    std = math.sqrt(variance)

    rms = math.sqrt(sum(sample**2 for sample in samples) / len(samples))

    cleaned_samples = remove_dc_offset(
        samples,
    )

    filtered_samples = bandpass_filter(
        cleaned_samples,
        sampling_rate,
    )

    frequencies, amplitudes = calculate_fft(
        filtered_samples,
        sampling_rate,
    )

    delta_power = calculate_band_power(
        frequencies,
        amplitudes,
        0.5,
        4.0,
    )

    theta_power = calculate_band_power(
        frequencies,
        amplitudes,
        4.0,
        8.0,
    )

    alpha_power = calculate_band_power(
        frequencies,
        amplitudes,
        8.0,
        13.0,
    )

    beta_power = calculate_band_power(
        frequencies,
        amplitudes,
        13.0,
        30.0,
    )

    dominant_wave = detect_dominant_wave(
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
    )

    attention_score = calculate_attention_score(
        theta_power,
        beta_power,
    )

    relaxation_score = calculate_relaxation_score(
        alpha_power,
        beta_power,
    )

    classification = classify_state(
        delta_power=delta_power,
        theta_power=theta_power,
        alpha_power=alpha_power,
        beta_power=beta_power,
        attention_score=attention_score,
        relaxation_score=relaxation_score,
    )

    predicted_state = classification["state"]
    prediction_confidence = classification["confidence"]

    return {
        "samples_count": len(samples),
        "average": round(average, 4),
        "minimum": min(samples),
        "maximum": max(samples),
        "std": round(std, 4),
        "rms": round(rms, 4),
        "delta_power": round(delta_power, 4),
        "theta_power": round(theta_power, 4),
        "alpha_power": round(alpha_power, 4),
        "beta_power": round(beta_power, 4),
        "dominant_wave": dominant_wave,
        "attention_score": attention_score,
        "relaxation_score": relaxation_score,
        "predicted_state": predicted_state,
        "prediction_confidence": prediction_confidence,
    }
