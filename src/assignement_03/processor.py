import numpy as np


def STFT_manual(signal, *, window_func_ind: str, window_size: int, overlap_size: int):
    """Manual STFT implementation with np FFT implementation"""
    frames = []

    hop_size = window_size - overlap_size

    match window_func_ind:
        case "Hamming":
            window_func = np.hamming(window_size)
        case "Hann":
            window_func = np.hanning(window_size)
        case "Blackman":
            window_func = np.blackman(window_size)
        case "Rectangular":
            window_func = np.ones(window_size)
        case _:
            raise ValueError("ex")

    for i in range(0, len(signal) - window_size, hop_size):
        segment = signal[i : i + window_size]
        windowed = segment * window_func

        spectrum = np.fft.fft(windowed)
        frames.append(np.abs(spectrum[: window_size // 2 + 1]))

    return np.array(frames).T


def STFT_manual_with_manual_DFT(
    signal, *, window_func_ind: str, window_size: int, overlap_size: int
):
    """Manual STFT implementation with manual DFT implementation"""
    frames = []

    hop_size = window_size - overlap_size

    match window_func_ind:
        case "Hamming":
            window_func = np.hamming(window_size)
        case "Hann":
            window_func = np.hanning(window_size)
        case "Blackman":
            window_func = np.blackman(window_size)
        case "Rectangular":
            window_func = np.ones(window_size)
        case _:
            raise ValueError("ex")

    N = window_size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    for i in range(0, len(signal) - window_size, hop_size):
        segment = signal[i : i + window_size]
        windowed = segment * window_func
        dft_segment = np.dot(e, windowed)
        half_spectrum = dft_segment[: N // 2 + 1]
        frames.append(np.abs(half_spectrum))

    return np.array(frames).T
