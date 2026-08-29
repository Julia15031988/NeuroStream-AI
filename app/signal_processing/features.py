import numpy as np


def calculate_fft(
    samples: list[float],
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.array([]), np.array([])

    signal = np.array(samples)

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1 / sampling_rate,
    )

    amplitudes = np.abs(fft_values)

    return frequencies, amplitudes


def calculate_band_power(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> float:
    if len(frequencies) == 0 or len(amplitudes) == 0:
        return 0.0

    mask = (frequencies >= low_freq) & (frequencies < high_freq)

    band_amplitudes = amplitudes[mask]

    if len(band_amplitudes) == 0:
        return 0.0

    return float(np.mean(band_amplitudes**2))
