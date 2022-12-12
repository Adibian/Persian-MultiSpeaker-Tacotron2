# raccoonML audio tools.
# MIT License
# Copyright (c) 2021 raccoonML (https://patreon.com/raccoonML)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software") to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR ANY OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import signal

_mel_basis = None


def load_wav(path, sr):
    # Loads an audio file and returns the waveform data.
    wav, _ = librosa.load(str(path), sr=sr)
    return wav


def save_wav(wav, path, sr):
    # Saves waveform data to audio file.
    sf.write(path, wav, sr)


def melspectrogram(wav, hparams):
    # Converts a waveform to a mel-scale spectrogram.
    # Output shape = (num_mels, frames)

    # Apply preemphasis
    if hparams.preemphasize:
        wav = preemphasis(wav, hparams.preemphasis)

    # Short-time Fourier Transform (STFT)
    D = librosa.stft(
        y=wav,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_size,
        win_length=hparams.win_size,
    )

    # Convert complex-valued output of STFT to absolute value (real)
    S = np.abs(D)

    # Build and cache mel basis
    # This improves speed when calculating thousands of mel spectrograms.
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)

    # Transform to mel scale
    S = np.dot(_mel_basis, S)

    # Dynamic range compression
    S = np.log(np.clip(S, a_min=1e-5, a_max=None))

    return S.astype(np.float32)


def inv_mel_spectrogram(S, hparams):
    # Converts a mel spectrogram to waveform using Griffin-Lim
    # Input shape = (num_mels, frames)

    # Denormalize
    S = np.exp(S)

    # Build and cache mel basis
    # This improves speed when calculating thousands of mel spectrograms.
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)

    # Inverse mel basis
    p = np.matmul(_mel_basis, _mel_basis.T)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    _inv_mel_basis = np.matmul(_mel_basis.T, np.diag(d))

    # Invert mel basis to recover linear spectrogram
    S = np.dot(_inv_mel_basis, S)

    # Use Griffin-Lim to recover waveform
    wav = _griffin_lim(S ** hparams.power, hparams)

    # Invert preemphasis
    if hparams.preemphasize:
        wav = inv_preemphasis(wav, hparams.preemphasis)

    return wav


def preemphasis(wav, k, preemphasize=True):
    # Amplifies high frequency content in a waveform.
    if preemphasize:
        wav = signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    # Inverts the preemphasis filter.
    if inv_preemphasize:
        wav = signal.lfilter([1], [1, -k], wav)
    return wav


def _build_mel_basis(hparams):
    return librosa.filters.mel(
        hparams.sample_rate,
        hparams.n_fft,
        n_mels=hparams.num_mels,
        fmin=hparams.fmin,
        fmax=hparams.fmax,
    )


def _griffin_lim(S, hparams):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S = np.abs(S).astype(np.complex)
    wav = librosa.istft(
        S * angles, hop_length=hparams.hop_size, win_length=hparams.win_size
    )
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(
            1j
            * np.angle(
                librosa.stft(
                    wav,
                    n_fft=hparams.n_fft,
                    hop_length=hparams.hop_size,
                    win_length=hparams.win_size,
                )
            )
        )
        wav = librosa.istft(
            S * angles, hop_length=hparams.hop_size, win_length=hparams.win_size
        )

    return wav
