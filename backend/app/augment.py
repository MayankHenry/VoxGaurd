"""
Telephone-channel augmentation.

VoxGuard targets phone-call audio, which is band-limited (300-3400 Hz), companded by the
G.711 codec and noisy. Applying the SAME degradation to real and synthetic clips during
training (a) makes the model robust to calls and (b) stops "clean studio vs. codec" from
being a shortcut that separates the classes.
"""
import zlib
import numpy as np
from scipy.signal import butter, sosfiltfilt

VARIANTS = ("none", "codec", "noisy")


def _peak(y):
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y / m if m > 1e-9 else y


def telephone_bandpass(y, sr=8000, low=300.0, high=3400.0):
    sos = butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def mulaw_roundtrip(y, mu=255):
    """G.711 style mu-law companding with 8-bit quantisation, then expansion."""
    y = np.clip(y, -1.0, 1.0)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((c + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
    return np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu


def add_noise(y, snr_db, rng):
    power = float(np.mean(y ** 2)) + 1e-12
    noise_power = power / (10 ** (snr_db / 10.0))
    return y + rng.normal(0.0, np.sqrt(noise_power), size=y.shape)


def apply_variant(y, variant, sr=8000, key=""):
    """Deterministic per (clip, variant): the same clip always gets the same degradation."""
    if variant == "none":
        return y
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}', expected one of {VARIANTS}")
    z = mulaw_roundtrip(_peak(telephone_bandpass(_peak(y), sr)))
    if variant == "noisy":
        rng = np.random.default_rng(zlib.crc32(f"{key}|{variant}".encode()))
        z = add_noise(z, snr_db=float(rng.uniform(15, 30)), rng=rng)
    return _peak(z)
