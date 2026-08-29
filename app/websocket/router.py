from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas.eeg import EEGPacket
from app.signal_processing.processor import process_eeg_samples

router = APIRouter()


@router.websocket("/ws/eeg")
async def eeg_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    print("✅ EEG client connected")

    try:
        while True:
            data = await websocket.receive_json()

            try:
                packet = EEGPacket.model_validate(data)
            except ValidationError as error:
                await websocket.send_json(
                    {
                        "status": "error",
                        "message": "Invalid EEG packet",
                        "details": error.errors(),
                    }
                )
                continue

            result = process_eeg_samples(
                packet.samples,
                packet.sampling_rate,
            )

            await websocket.send_json(
                {
                    "status": "processed",
                    "channel": packet.channel,
                    "sampling_rate": packet.sampling_rate,
                    "result": result,
                }
            )

    except WebSocketDisconnect:
        print("❌ EEG client disconnected")
