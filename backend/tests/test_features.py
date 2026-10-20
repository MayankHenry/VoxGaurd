"""Pipeline sanity tests for feature extraction (uses synthetic tones, not model accuracy)."""
import numpy as np
import pytest
import soundfile as sf
from app.features import extract_features, FEATURE_NAMES


def _tone(path, freq=150.0, sr=16000, dur=2.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sf.write(path, 0.3 * np.sin(2 * np.pi * (freq + 5 * np.sin(2 * np.pi * 4 * t)) * t), sr)


def test_extract_features_shape_and_sanity(tmp_path):
    p = tmp_path / "tone.wav"
    _tone(p)
    f = extract_features(str(p))
    v = f.to_vector()
    assert v.shape[0] == len(FEATURE_NAMES) == 37
    assert not np.isnan(v).any()
    assert f.duration_sec > 1.9 and f.pitch_mean > 0


def test_same_content_different_sample_rate_gives_similar_features(tmp_path):
    """Telephone-band normalisation: sample rate of the source must not leak into features."""
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _tone(a, sr=16000)
    _tone(b, sr=44100)
    va, vb = extract_features(str(a)).to_vector(), extract_features(str(b)).to_vector()
    assert np.corrcoef(va, vb)[0, 1] > 0.99


def test_silence_raises(tmp_path):
    p = tmp_path / "s.wav"
    sf.write(p, np.zeros(16000), 16000)
    with pytest.raises(RuntimeError):
        extract_features(str(p))
