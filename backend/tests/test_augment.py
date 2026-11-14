import numpy as np
import pytest
from app.augment import apply_variant, mulaw_roundtrip, telephone_bandpass, add_noise

SR = 8000


def _tone(f, dur=1.0, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    return amp * np.sin(2 * np.pi * f * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_bandpass_keeps_voice_band_and_removes_edges():
    keep = _rms(telephone_bandpass(_tone(1000))) / _rms(_tone(1000))
    low = _rms(telephone_bandpass(_tone(50))) / _rms(_tone(50))
    high = _rms(telephone_bandpass(_tone(3900))) / _rms(_tone(3900))
    assert keep > 0.9 and low < 0.1 and high < 0.3


def test_mulaw_is_close_to_identity_but_quantised():
    y = _tone(500)
    z = mulaw_roundtrip(y)
    assert np.corrcoef(y, z)[0, 1] > 0.999
    assert not np.allclose(y, z)


def test_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    y = _tone(300)
    n = add_noise(y, 20, rng) - y
    snr = 10 * np.log10(np.mean(y ** 2) / np.mean(n ** 2))
    assert 18 < snr < 22


def test_variants_are_deterministic_and_none_is_identity():
    y = _tone(400)
    assert np.array_equal(apply_variant(y, "none"), y)
    a, b = apply_variant(y, "noisy", key="clip1"), apply_variant(y, "noisy", key="clip1")
    c = apply_variant(y, "noisy", key="clip2")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        apply_variant(_tone(400), "bogus")
