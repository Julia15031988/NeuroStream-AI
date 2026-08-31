import math
import random

import torch
from torch.utils.data import Dataset

STATE_TO_LABEL = {
    "relaxed": 0,
    "focused": 1,
    "drowsy": 2,
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

        state_frequencies = {
            "relaxed": 10.0,
            "focused": 20.0,
            "drowsy": 6.0,
        }

        for state, frequency in state_frequencies.items():
            for _ in range(samples_per_state):
                signal = self._generate_signal(
                    frequency=frequency,
                    signal_length=signal_length,
                    sampling_rate=sampling_rate,
                )

                self.data.append(signal)
                self.labels.append(STATE_TO_LABEL[state])

    @staticmethod
    def _generate_signal(
        frequency: float,
        signal_length: int,
        sampling_rate: int,
    ) -> torch.Tensor:
        phase = random.uniform(0.0, 2.0 * math.pi)

        values = []

        for index in range(signal_length):
            time = index / sampling_rate

            eeg_value = math.sin(
                2.0 * math.pi * frequency * time + phase
            )

            noise = random.gauss(0.0, 0.15)

            values.append(eeg_value + noise)

        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], self.labels[index]
