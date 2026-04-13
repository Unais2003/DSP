from blinker import signal
import numpy as np


def get_sampling_rate(df) -> float:
    """
    Calculates the sampling rate based on amount of samples and
    last observed time.
    """
    N = df.shape[0]
    L = df.iloc[-1, :]["time_s"]

    fs = (N - 1) / L
    return fs


def DFT_batched(*, signal, start, end) -> tuple:
    fs = 256

    start = int(start * fs)
    end = int(end * fs)

    N = end - start
    signal = signal["signal"].values[start:end]

    n = np.arange(N)
    fk = np.arange(N).reshape((N, 1))
    e = np.exp(-2j * np.pi * (n / N) * fk)

    dft = np.dot(e, signal)

    amp = np.abs(dft)
    freq = np.arange(N) * (fs / N)

    return (amp, freq)


def downsample_signal(df, original_fs, target_fs):
    """
    Downsample signal WITHOUT filtering.
    Uses integer factor (power-of-2 friendly).
    """
    factor = max(1, int(original_fs / target_fs))
    downsampled_df = df.iloc[::factor].reset_index(drop=True)
    new_fs = original_fs / factor

    return downsampled_df, new_fs, factor


def DFT_batched_downsampled(*, signal, start, end, fs):
    """
    DFT for downsampled signal (dynamic sampling rate)
    """
    start_idx = int(start * fs)
    end_idx = int(end * fs)

    start = max(0, int(start * fs))
    end = min(len(signal), int(end * fs))

    signal_segment = signal["signal"].values[start_idx:end_idx]
    N = len(signal_segment)

    if N == 0:
        return np.array([]), np.array([])

    n = np.arange(N)
    k = np.arange(N).reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)
    dft = np.dot(e, signal_segment)

    amp = np.abs(dft)
    freq = np.arange(N) * (fs / N)

    return amp, freq
