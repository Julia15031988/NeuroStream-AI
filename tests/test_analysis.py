from app.signal_processing.analysis import (
    calculate_attention_score,
    calculate_relaxation_score,
    detect_dominant_wave,
)


def test_detect_dominant_wave() -> None:
    result = detect_dominant_wave(
        delta_power=1.0,
        theta_power=2.0,
        alpha_power=10.0,
        beta_power=3.0,
    )

    assert result == "alpha"


def test_calculate_attention_score() -> None:
    result = calculate_attention_score(
        theta_power=25.0,
        beta_power=75.0,
    )

    assert result == 75.0


def test_calculate_relaxation_score() -> None:
    result = calculate_relaxation_score(
        alpha_power=80.0,
        beta_power=20.0,
    )

    assert result == 80.0


def test_attention_score_with_zero_power() -> None:
    result = calculate_attention_score(
        theta_power=0.0,
        beta_power=0.0,
    )

    assert result == 0.0


def test_relaxation_score_with_zero_power() -> None:
    result = calculate_relaxation_score(
        alpha_power=0.0,
        beta_power=0.0,
    )

    assert result == 0.0
