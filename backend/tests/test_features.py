"""
Day 1 smoke test for the feature extraction pipeline.

We don't have real ASVspoof samples loaded yet (that's a Day 2 data
acquisition task), so this generates a synthetic sine-wave "voice-like"
signal purely to prove the extraction code runs end-to-end without
errors and returns sane, non-NaN values. This is NOT a model accuracy
test — just a pipeline sanity check.
"""
import numpy as np
import soundfile as sf
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.features import extract_features


def _make_test_tone(path, freq=150.0, sr=16000, duration=2.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # simple vibrato to mimic natural pitch variation
    vibrato = 5 * np.sin(2 * np.pi * 4 * t)
    signal = 0.3 * np.sin(2 * np.pi * (freq + vibrato) * t)
    sf.write(path, signal, sr)


def test_extract_features_runs_and_returns_sane_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_tone.wav")
        _make_test_tone(path)

        features = extract_features(path)

        assert features.duration_sec > 1.9
        assert len(features.mfcc_mean) == 13
        assert len(features.mfcc_std) == 13
        assert not np.isnan(features.spectral_flatness_mean)
        assert not np.isnan(features.spectral_centroid_mean)
        assert features.pitch_mean > 0  # should detect our synthetic pitch

        vector = features.to_vector()
        assert vector.shape[0] == 13 + 13 + 5  # mfcc_mean + mfcc_std + 5 scalars
        assert not np.isnan(vector).any()


def test_extract_features_raises_on_silence():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "silence.wav")
        sr = 16000
        silence = np.zeros(sr * 1)
        sf.write(path, silence, sr)

        try:
            extract_features(path)
            assert False, "Expected RuntimeError on silent audio"
        except RuntimeError:
            pass


if __name__ == "__main__":
    test_extract_features_runs_and_returns_sane_values()
    test_extract_features_raises_on_silence()
    print("All Day 1 feature extraction tests passed.")
