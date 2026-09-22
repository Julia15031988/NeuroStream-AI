import math
import random

import torch
from torch.utils.data import Dataset

STATE_TO_LABEL = {
    "relaxed": 0,
    "focused": 1,
    "drowsy": 2,
}

STATE_FREQUENCY_RANGES = {
    "relaxed": (8.0, 13.0),
    "focused": (13.0, 30.0),
    "drowsy": (4.0, 8.0),
}


class SyntheticEEGDataset(Dataset):
    def __init__(
        self,
        samples_per_state: int = 1000,
        signal_length: int = 256,
        sampling_rate: int = 256,
    ) -> None:
        self.data: list[torch.Tensor] = []
        self.labels: list[int] = []

        for state, frequency_range in STATE_FREQUENCY_RANGES.items():
            for _ in range(samples_per_state):
                signal = self._generate_signal(
                    frequency_range=frequency_range,
                    signal_length=signal_length,
                    sampling_rate=sampling_rate,
                )

                self.data.append(signal)
                self.labels.append(STATE_TO_LABEL[state])

    @staticmethod
    def _generate_signal(
        frequency_range: tuple[float, float],
        signal_length: int,
        sampling_rate: int,
    ) -> torch.Tensor:
        dominant_frequency = random.uniform(
            frequency_range[0],
            frequency_range[1],
        )

        secondary_frequency = random.uniform(
            4.0,
            30.0,
        )

        phase = random.uniform(
            0.0,
            2.0 * math.pi,
        )

        secondary_phase = random.uniform(
            0.0,
            2.0 * math.pi,
        )

        dominant_amplitude = random.uniform(
            0.6,
            1.4,
        )

        secondary_amplitude = random.uniform(
            0.05,
            0.35,
        )

        noise_std = random.uniform(
            0.1,
            0.4,
        )

        dc_offset = random.uniform(
            -0.3,
            0.3,
        )

        values = []

        for index in range(signal_length):
            time = index / sampling_rate

            dominant_wave = dominant_amplitude * math.sin(
                2.0
                * math.pi
                * dominant_frequency
                * time
                + phase
            )

            secondary_wave = secondary_amplitude * math.sin(
                2.0
                * math.pi
                * secondary_frequency
                * time
                + secondary_phase
            )

            noise = random.gauss(
                0.0,
                noise_std,
            )

            eeg_value = (
                dominant_wave
                + secondary_wave
                + noise
                + dc_offset
            )

            values.append(eeg_value)

        return torch.tensor(
            values,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        return self.data[index], self.labels[index]
