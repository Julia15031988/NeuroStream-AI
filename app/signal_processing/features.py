import numpy as np
from scipy.signal import welch


def calculate_psd(
    samples: list[float],
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.array([]), np.array([])

    signal = np.asarray(
        samples,
        dtype=float,
    )

    frequencies, power_spectral_density = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(
            len(signal),
            256,
        ),
    )

    return frequencies, power_spectral_density


def calculate_band_power(
    frequencies: np.ndarray,
    power_spectral_density: np.ndarray,
    low_freq: float,
    high_freq: float,
) -> float:
    if (
        len(frequencies) == 0
        or len(power_spectral_density) == 0
    ):
        return 0.0

    mask = (
        (frequencies >= low_freq)
        & (frequencies < high_freq)
    )

    band_frequencies = frequencies[mask]
    band_psd = power_spectral_density[mask]

    if len(band_psd) == 0:
        return 0.0

    return float(
        np.trapezoid(
            band_psd,
            band_frequencies,
        )
    )

