"""
Day 4: Intake pipeline for freshly recorded real-speech samples.

Problem this solves: recordings straight off a phone/laptop mic come
in inconsistent formats (m4a, mp3, wav at various sample rates),
sometimes with a dead/corrupt file or an accidental near-silent
recording mixed in. Dropping these directly into data/raw/real/ risks
the manifest builder choking on one bad file, or the classifier
learning "16kHz vs 44kHz" as a fake real-vs-synthetic signal instead
of an actual voice-cloning artifact.

This script:
1. Scans a staging folder (wherever you dropped your raw recordings)
2. Validates each file (loads OK, not silent, reasonable duration)
3. Resamples everything to a consistent 16kHz mono WAV (matching what
   scripts/generate_synthetic_samples.py already produces, so real
   and synthetic samples aren't distinguishable by sample rate alone)
4. Renames systematically and copies into data/raw/real/
5. Prints a summary + flags anything skipped and why

Usage:
    python scripts/prepare_real_samples.py --input /path/to/raw_recordings --speaker mayank
"""
import argparse
import sys
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
REAL_DIR = BACKEND_DIR / "data" / "raw" / "real"
REAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
SILENCE_AMPLITUDE_THRESHOLD = 1e-4
SUPPORTED_INPUT_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def validate_and_convert(src_path: Path):
    """
    Returns (audio_array, sample_rate) on success, or raises ValueError
    with a human-readable reason on failure.
    """
    try:
        y, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        raise ValueError(f"could not be loaded/decoded ({e})")

    if y.size == 0:
        raise ValueError("is empty")

    duration = len(y) / TARGET_SR
    if duration < MIN_DURATION_SEC:
        raise ValueError(f"too short ({duration:.2f}s, minimum {MIN_DURATION_SEC}s)")
    if duration > MAX_DURATION_SEC:
        raise ValueError(f"too long ({duration:.2f}s, maximum {MAX_DURATION_SEC}s) — trim it first")

    if np.max(np.abs(y)) < SILENCE_AMPLITUDE_THRESHOLD:
        raise ValueError("appears silent (max amplitude below threshold)")

    return y, TARGET_SR


def run(input_dir: Path, speaker_tag: str):
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        sys.exit(1)

    candidates = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )

    if not candidates:
        print(f"No supported audio files found in {input_dir} "
              f"(looking for {sorted(SUPPORTED_INPUT_EXTENSIONS)})")
        sys.exit(1)

    print(f"Found {len(candidates)} candidate file(s) in {input_dir}\n")

    accepted = []
    rejected = []

    # find the next available index so re-runs don't overwrite existing files
    existing = list(REAL_DIR.glob(f"real_{speaker_tag}_*.wav"))
    start_idx = len(existing) + 1

    for i, src in enumerate(candidates, start=start_idx):
        try:
            y, sr = validate_and_convert(src)
        except ValueError as reason:
            print(f"SKIPPED  {src.name}: {reason}")
            rejected.append((src.name, str(reason)))
            continue

        out_name = f"real_{speaker_tag}_{i:03d}.wav"
        out_path = REAL_DIR / out_name
        sf.write(out_path, y, sr)
        print(f"ACCEPTED {src.name} -> {out_name}  ({len(y)/sr:.2f}s)")
        accepted.append(out_name)

    print(f"\nSummary: {len(accepted)} accepted, {len(rejected)} rejected out of {len(candidates)} total.")
    if rejected:
        print("Rejected files and reasons:")
        for name, reason in rejected:
            print(f"  - {name}: {reason}")
    print(f"\nAccepted files copied to: {REAL_DIR}")
    print("Next step: python scripts/build_manifest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder containing your raw recordings")
    parser.add_argument("--speaker", required=True, help="Short tag for who recorded these (e.g. mayank, radhika)")
    args = parser.parse_args()

    run(Path(args.input), args.speaker.lower().strip())
