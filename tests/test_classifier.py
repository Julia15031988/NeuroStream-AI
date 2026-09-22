from app.ml.classifier import classify_state


def test_classify_state_returns_valid_result() -> None:
    result = classify_state(
        delta_power=0.1,
        theta_power=0.2,
        alpha_power=0.6,
        beta_power=0.1,
        attention_score=20.0,
        relaxation_score=80.0,
    )

    assert "state" in result
    assert "confidence" in result
    assert result["state"] in {
        "relaxed",
        "focused",
        "drowsy",
    }
    assert 0.0 <= result["confidence"] <= 1.0

def test_classify_relaxed_state() -> None:
    result = classify_state(
        delta_power=0.05,
        theta_power=0.10,
        alpha_power=0.75,
        beta_power=0.10,
        attention_score=15.0,
        relaxation_score=88.0,
    )

    assert result["state"] == "relaxed"
    assert 0.0 <= result["confidence"] <= 1.0

def test_classify_focused_state() -> None:
    result = classify_state(
        delta_power=0.05,
        theta_power=0.10,
        alpha_power=0.10,
        beta_power=0.75,
        attention_score=88.0,
        relaxation_score=15.0,
    )

    assert result["state"] == "focused"
    assert 0.0 <= result["confidence"] <= 1.0

def test_classify_drowsy_state() -> None:
    result = classify_state(
        delta_power=0.10,
        theta_power=0.75,
        alpha_power=0.10,
        beta_power=0.05,
        attention_score=12.0,
        relaxation_score=20.0,
    )

    assert result["state"] == "drowsy"
    assert 0.0 <= result["confidence"] <= 1.0
