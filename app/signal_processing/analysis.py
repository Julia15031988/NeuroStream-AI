def detect_dominant_wave(
    delta_power: float,
    theta_power: float,
    alpha_power: float,
    beta_power: float,
) -> str:
    bands = {
        "delta": delta_power,
        "theta": theta_power,
        "alpha": alpha_power,
        "beta": beta_power,
    }

    return max(
        bands,
        key=bands.get,
    )


def calculate_attention_score(
    theta_power: float,
    beta_power: float,
) -> float:
    total_power = theta_power + beta_power

    if total_power == 0:
        return 0.0

    score = beta_power / total_power * 100

    return round(
        min(max(score, 0.0), 100.0),
        2,
    )


def calculate_relaxation_score(
    alpha_power: float,
    beta_power: float,
) -> float:
    total_power = alpha_power + beta_power

    if total_power == 0:
        return 0.0

    score = alpha_power / total_power * 100

    return round(
        min(max(score, 0.0), 100.0),
        2,
    )
