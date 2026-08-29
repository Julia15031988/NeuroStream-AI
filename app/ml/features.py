def build_feature_vector(
    delta_power: float,
    theta_power: float,
    alpha_power: float,
    beta_power: float,
    attention_score: float,
    relaxation_score: float,
) -> list[float]:
    total_power = delta_power + theta_power + alpha_power + beta_power

    if total_power == 0:
        return [0.0] * 6

    return [
        delta_power / total_power,
        theta_power / total_power,
        alpha_power / total_power,
        beta_power / total_power,
        attention_score / 100,
        relaxation_score / 100,
    ]
