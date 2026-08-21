"""
Audio feature extraction for VoxGuard.

Day 1 scope: get a clean, tested feature-extraction function working
on a single audio file. This is the foundation the Day 4-7 classifier
training will consume — so it's built and unit-tested first, before
any model code, to avoid debugging both at once later.

Features extracted (grounded in known TTS/voice-clone artifacts):
- MFCCs (13 coefficients + deltas): captures timbre/spectral envelope
- Spectral flatness: synthetic speech tends to be spectrally "flatter"
- Pitch (f0) stability: real speech has natural micro-variation in
  pitch; many TTS systems produce unnaturally smooth f0 contours
- Spectral centroid & bandwidth: cloned voices often show subtly
  different high-frequency energy distribution
"""
from dataclasses import dataclass, asdict
import numpy as np
import librosa


@dataclass
class AudioFeatures:
    mfcc_mean: list
    mfcc_std: list
    spectral_flatness_mean: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    pitch_mean: float
    pitch_std: float
    duration_sec: float

    def to_vector(self) -> np.ndarray:
        """Flatten into a single feature vector for classifier input."""
        return np.concatenate([
            self.mfcc_mean,
            self.mfcc_std,
            [
                self.spectral_flatness_mean,
                self.spectral_centroid_mean,
                self.spectral_bandwidth_mean,
                self.pitch_mean,
                self.pitch_std,
            ],
        ]).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features(file_path: str, sr: int = 16000, n_mfcc: int = 13) -> AudioFeatures:
    """
    Load an audio file and extract the feature set used for
    real-vs-synthetic classification.

    Raises:
        RuntimeError if the file can't be loaded or is silent/empty.
    """
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{file_path}': {e}")

    if y.size == 0 or np.max(np.abs(y)) < 1e-6:
        raise RuntimeError(f"Audio file '{file_path}' is empty or silent.")

    duration_sec = len(y) / loaded_sr

    mfcc = librosa.feature.mfcc(y=y, sr=loaded_sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=loaded_sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=loaded_sr)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.nanmean(voiced_f0)) if voiced_f0.size > 0 else 0.0
    pitch_std = float(np.nanstd(voiced_f0)) if voiced_f0.size > 0 else 0.0

    return AudioFeatures(
        mfcc_mean=mfcc_mean.tolist(),
        mfcc_std=mfcc_std.tolist(),
        spectral_flatness_mean=float(np.mean(spectral_flatness)),
        spectral_centroid_mean=float(np.mean(spectral_centroid)),
        spectral_bandwidth_mean=float(np.mean(spectral_bandwidth)),
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        duration_sec=float(duration_sec),
    )
