"""
Day 2: Synthetic (AI-generated) speech sample generation.

We generate the "synthetic" class ourselves using espeak-ng, an
offline TTS engine, varying voice, pitch, and speaking rate to build
diversity, since espeak-ng runs fully offline (no dataset-hosting
domains reachable from this environment).

NOTE ON THE "real" CLASS: this script only produces the synthetic
side of the dataset. The real-human-speech side must come from either
(a) recordings the team makes themselves, or (b) a public corpus such
as ASVspoof2019 / VCTK downloaded on a machine with normal internet
access (see data/raw/real/README.md for exact instructions). This
sandbox's network is restricted to a small domain allowlist that does
not include Zenodo/OpenSLR/Kaggle, so it cannot fetch those directly.

Usage:
    python scripts/generate_synthetic_samples.py
"""
import subprocess
import itertools
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A small varied sentence pool — deliberately includes different
# sentence lengths/structures so extracted features aren't confounded
# by text content alone.
SENTENCES = [
    "Hello, this is a verification call regarding your bank account.",
    "Please confirm your identity by providing your date of birth.",
    "The quarterly results will be announced next Monday morning.",
    "I need you to transfer the funds immediately, it is urgent.",
    "Can you hear me clearly on this line right now?",
    "Your package could not be delivered due to an incorrect address.",
    "We are calling from the technical support department today.",
    "This message is to remind you about your upcoming appointment.",
    "The weather today is expected to be partly cloudy with rain.",
    "Thank you for calling, your call is important to us.",
]

# espeak-ng built-in voice variants (offline, no downloads needed)
VOICES = ["en", "en-us", "en-gb"]
PITCHES = [30, 50, 70]   # espeak-ng pitch range 0-99
RATES = [140, 175, 210]  # words per minute


def generate_all():
    manifest = []
    combos = list(itertools.product(VOICES, PITCHES, RATES))
    sample_idx = 0

    for sentence in SENTENCES:
        # not every combo x every sentence (too many); rotate combos per sentence
        voice, pitch, rate = combos[sample_idx % len(combos)]
        sample_idx += 1

        filename = f"synth_{sample_idx:03d}_{voice}_p{pitch}_r{rate}.wav"
        out_path = OUTPUT_DIR / filename

        cmd = [
            "espeak-ng", sentence,
            "-v", voice,
            "-p", str(pitch),
            "-s", str(rate),
            "-w", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED: {filename} -> {result.stderr}")
            continue

        manifest.append({
            "filename": filename,
            "label": "synthetic",
            "source": "espeak-ng",
            "voice": voice,
            "pitch": pitch,
            "rate_wpm": rate,
            "text": sentence,
        })
        print(f"Generated: {filename}")

    return manifest


if __name__ == "__main__":
    manifest = generate_all()
    print(f"\nDone. Generated {len(manifest)} synthetic samples in {OUTPUT_DIR}")
