import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_eeg_websocket_processes_valid_packet() -> None:
    sampling_rate = 256

    samples = [
        math.sin(
            2 * math.pi * 10 * i / sampling_rate
        )
        for i in range(256)
    ]

    with client.websocket_connect("/ws/eeg") as websocket:
        websocket.send_json(
            {
                "channel": "Fp1",
                "sampling_rate": sampling_rate,
                "samples": samples,
            }
        )

        response = websocket.receive_json()

    assert response["status"] == "processed"
    assert response["channel"] == "Fp1"
    assert response["sampling_rate"] == 256

    result = response["result"]

    assert result["samples_count"] == 256
    assert result["dominant_wave"] == "alpha"
    assert result["predicted_state"] in {
        "relaxed",
        "focused",
        "drowsy",
    }
    assert 0.0 <= result["prediction_confidence"] <= 1.0

def test_eeg_websocket_rejects_invalid_packet() -> None:
    with client.websocket_connect("/ws/eeg") as websocket:
        websocket.send_json(
            {
                "channel": "Fp1",
                "sampling_rate": 256,
            }
        )

        response = websocket.receive_json()

    assert response["status"] == "error"
    assert response["message"] == "Invalid EEG packet"
    assert "details" in response
