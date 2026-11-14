"""Tests for the data-import / evaluation tooling (no real datasets needed)."""
import random
import numpy as np
import soundfile as sf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_NAMES
from scripts.import_asvspoof import import_asvspoof, parse_protocol, balanced_pick
from scripts.generate_neural_samples import convert_to_wav16k, out_name
from scripts.evaluate_external import evaluate_dirs
from scripts.build_manifest import parse


def _flac(path, f=200, dur=1.5, sr=16000):
    t = np.arange(int(sr * dur)) / sr
    sf.write(path, 0.3 * np.sin(2 * np.pi * f * t) + 0.02 * np.random.randn(len(t)), sr)


def _fake_asvspoof(tmp_path):
    audio = tmp_path / "flac"
    audio.mkdir()
    lines = []
    for spk in range(3):
        for i in range(4):
            fid = f"LA_T_{spk}{i:03d}"
            _flac(audio / f"{fid}.flac")
            lines.append(f"LA_00{spk} {fid} - - bonafide")
    for atk in ("A01", "A02"):
        for i in range(5):
            fid = f"LA_T_{atk}{i}"
            _flac(audio / f"{fid}.flac", f=300)
            lines.append(f"LA_009 {fid} - {atk} spoof")
    proto = tmp_path / "proto.txt"
    proto.write_text("\n".join(lines))
    return proto, audio


def test_protocol_parse_and_balanced_pick(tmp_path):
    proto, _ = _fake_asvspoof(tmp_path)
    rows = parse_protocol(proto)
    assert len(rows) == 22
    picked = balanced_pick([r for r in rows if r["label"] == "bonafide"], "speaker", 6, random.Random(0))
    assert sorted({r["speaker"] for r in picked}) == ["LA_000", "LA_001", "LA_002"]


def test_import_names_follow_manifest_convention(tmp_path):
    proto, audio = _fake_asvspoof(tmp_path)
    res = import_asvspoof(proto, audio, tmp_path / "r", tmp_path / "f", n_bonafide=6, n_spoof=6)
    assert res["real"] == 6 and res["spoof"] == 6 and res["missing"] == 0
    real = sorted((tmp_path / "r").iterdir())[0]
    fake = sorted((tmp_path / "f").iterdir())[0]
    assert parse(real, "real")[0] == "human" and parse(real, "real")[1].startswith("LA00")
    fam, variant, group = parse(fake, "synthetic")
    assert fam in {"asvA01", "asvA02"} and variant in {"A01", "A02"} and group.startswith("LA_T_")
    assert sf.info(real).samplerate == 16000


def test_neural_conversion_and_naming(tmp_path):
    src = tmp_path / "x.mp3"
    sf.write(src, 0.3 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000), 24000, format="MP3")
    dst = tmp_path / "y.wav"
    convert_to_wav16k(src, dst)
    assert sf.info(dst).samplerate == 16000 and sf.info(dst).channels == 1
    assert out_name("synth", "aria", "LJ001-0001") == "synth_edgetts_aria_LJ001-0001.wav"
    assert parse(tmp_path / out_name("synth", "aria", "LJ001-0001"), "synthetic") == ("edgetts", "aria", "LJ001-0001")


def test_evaluate_external_reports_metrics(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = np.array(["real"] * 20 + ["synthetic"] * 20)
    sc = StandardScaler().fit(X)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(sc.transform(X), y)
    mp = tmp_path / "m.joblib"
    joblib.dump({"model": rf, "scaler": sc, "feature_idx": list(range(len(FEATURE_NAMES))), "version": "t"}, mp)
    (tmp_path / "r").mkdir(); (tmp_path / "f").mkdir()
    for i in range(4):
        _flac(tmp_path / "r" / f"r{i}.flac", f=150 + 10 * i)
        _flac(tmp_path / "f" / f"f{i}.flac", f=300 + 10 * i)
    res = evaluate_dirs(tmp_path / "r", tmp_path / "f", variants=("none", "codec"), model_path=mp)
    assert res["n_real"] == 4 and res["n_fake"] == 4 and set(res["by_variant"]) == {"none", "codec"}
    assert "eer" in res["by_variant"]["codec"]
