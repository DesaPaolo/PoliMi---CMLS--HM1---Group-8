import numpy as np
import librosa
import scipy as sp


# Feature Extraction

# Zero Crossing Rate
def compute_zcr(win, Fs):
    win_sign = np.sign(win)
    N = win.shape[0]
    sign_diff = np.abs(win_sign[:-1] - win_sign[1:])
    zcr = (Fs / (2 * N)) * np.sum(sign_diff)
    return zcr


# Spectral Centroid
def compute_speccentr(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / np.sum(np.abs(spec))
    return centr


# Spectral Decrease
def compute_specdec(spec):
    mul_fact = 1 / np.sum(np.abs(spec[1:]))
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1)
    den = np.arange(1, len(spec))
    spectral_decrease = mul_fact * np.sum(num / den)
    return spectral_decrease


# Mel Frequency Cepstrum Coefficients (MFCCs)
def compute_mfcc(audio, fs, n_mfcc):
    X = np.abs(librosa.stft(
        audio,
        window='hamming',
        n_fft=1024,
        hop_length=512, )
    )

    mel = librosa.filters.mel(
        sr=fs,
        n_fft=1024,
        n_mels=40,
        fmin=133.33,
        fmax=6853.8
    )

    melspectrogram = np.dot(mel, X)
    log_melspectrogram = np.log10(melspectrogram + 1e-16)
    mfcc = sp.fftpack.dct(log_melspectrogram, axis=0, norm='ortho')[1:n_mfcc + 1]
    return mfcc
