"""
Fetch extra REAL human speakers (different voices / mics / one French speaker) so the
real class is not dominated by a single LJSpeech voice.

Source: public test samples in speechbrain/speechbrain (tests/samples/ASR, tests/samples/lang).
Adds clips named real_<speakerTag>_<id>.wav  (speakerTags: sbspk1, sbspk2, sbeng1, sbeng5, sbfr2).

This is still a tiny corpus (~5 speakers). It exists to expose / reduce speaker-channel
confounds in development. For final results add ASVspoof2019 bonafide (many speakers) and
your own recordings.

Usage:  python scripts/fetch_extra_real_speakers.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from scripts.prepare_real_samples import validate_and_convert
import soundfile as sf

REPO = "https://github.com/speechbrain/speechbrain"
PLAN = [  # (relative glob, speaker tag)
    ("tests/samples/ASR/spk1_*.wav", "sbspk1"),
    ("tests/samples/ASR/spk2_*.wav", "sbspk2"),
    ("tests/samples/lang/English/example1.wav", "sbeng1"),
    ("tests/samples/lang/English/example5.wav", "sbeng5"),
    ("tests/samples/lang/French/example2.flac", "sbfr2"),
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sb_"))
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(tmp / "r")],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp / "r"), "sparse-checkout", "set", "tests/samples"],
                       check=True, capture_output=True)
        for tag in {t for _, t in PLAN}:
            for old in config.REAL_DIR.glob(f"real_{tag}_*.wav"):
                old.unlink()
        kept = 0
        for pattern, tag in PLAN:
            for src in sorted((tmp / "r").glob(pattern)):
                try:
                    y, sr = validate_and_convert(src)
                except ValueError as e:
                    print(f"SKIPPED {src.name}: {e}")
                    continue
                sf.write(config.REAL_DIR / f"real_{tag}_{src.stem}.wav", y, sr)
                kept += 1
        print(f"Extra real clips saved: {kept} -> {config.REAL_DIR}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
