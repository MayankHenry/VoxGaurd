"""
Audio feature extraction for VoxGuard.

Design decisions (each one removes a way a classifier could "cheat"):
- Audio is loaded at FEATURE_SR = 8 kHz ("telephone band"). VoxGuard targets
  phone-call audio, and this also guarantees a 22 kHz studio recording and an
  8 kHz TTS file have identical bandwidth, so bandwidth/sample-rate can't be
  used as a shortcut for real-vs-synthetic.
- Signal is peak-normalised so loudness differences can't leak into MFCC0 etc.
- Features are chosen from known TTS/voice-clone artefacts: spectral
  smoothness (flatness), pitch micro-variation (jitter proxy), dynamic range
  (RMS std), voiced ratio, plus MFCC timbre statistics.
"""
from dataclasses import dataclass, asdict
import numpy as np
import librosa

FEATURE_SR = 8000
N_MFCC = 13

SCALAR_NAMES = [
    "spectral_flatness_mean", "spectral_centroid_mean", "spectral_bandwidth_mean",
    "pitch_mean", "pitch_std",
    "spectral_rolloff_mean", "zcr_mean", "rms_mean", "rms_std",
    "voiced_ratio", "f0_jitter",
]
FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(N_MFCC)]
    + SCALAR_NAMES
)


@dataclass
class AudioFeatures:
    mfcc_mean: list
    mfcc_std: list
    spectral_flatness_mean: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    pitch_mean: float
    pitch_std: float
    spectral_rolloff_mean: float
    zcr_mean: float
    rms_mean: float
    rms_std: float
    voiced_ratio: float
    f0_jitter: float
    duration_sec: float

    def to_vector(self) -> np.ndarray:
        scalars = [getattr(self, n) for n in SCALAR_NAMES]
        return np.concatenate([self.mfcc_mean, self.mfcc_std, scalars]).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features(file_path: str, sr: int = FEATURE_SR, n_mfcc: int = N_MFCC) -> AudioFeatures:
    """Load an audio file and compute the feature set. Raises RuntimeError on bad audio."""
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{file_path}': {e}")

    if y.size == 0 or np.max(np.abs(y)) < 1e-6:
        raise RuntimeError(f"Audio file '{file_path}' is empty or silent.")

    y = y / np.max(np.abs(y))  # peak-normalise
    duration_sec = len(y) / loaded_sr

    mfcc = librosa.feature.mfcc(y=y, sr=loaded_sr, n_mfcc=n_mfcc)
    flat = librosa.feature.spectral_flatness(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=loaded_sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=loaded_sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=loaded_sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=loaded_sr)
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0
    vf0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    vf0 = vf0[~np.isnan(vf0)]
    if vf0.size > 2:
        pitch_mean, pitch_std = float(np.mean(vf0)), float(np.std(vf0))
        f0_jitter = float(np.mean(np.abs(np.diff(vf0))) / max(pitch_mean, 1e-6))
    else:
        pitch_mean = pitch_std = f0_jitter = 0.0

    return AudioFeatures(
        mfcc_mean=np.mean(mfcc, axis=1).tolist(),
        mfcc_std=np.std(mfcc, axis=1).tolist(),
        spectral_flatness_mean=float(np.mean(flat)),
        spectral_centroid_mean=float(np.mean(cent)),
        spectral_bandwidth_mean=float(np.mean(bw)),
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        spectral_rolloff_mean=float(np.mean(roll)),
        zcr_mean=float(np.mean(zcr)),
        rms_mean=float(np.mean(rms)),
        rms_std=float(np.std(rms)),
        voiced_ratio=voiced_ratio,
        f0_jitter=f0_jitter,
        duration_sec=float(duration_sec),
    )
