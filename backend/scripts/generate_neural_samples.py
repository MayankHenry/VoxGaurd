"""
Generate NEURAL text-to-speech clips (modern AI voices) with the free `edge-tts` package
(Microsoft Edge online voices; needs internet, no API key).

    pip install edge-tts
    python scripts/generate_neural_samples.py                    # all transcripts x 4 voices
    python scripts/generate_neural_samples.py --limit 10 --voices aria,neerja

Writes synth_edgetts_<voice>_<uttid>.wav (16 kHz mono) into data/raw/synthetic so the file
becomes a 4th generator family ("edgetts") for leave-one-engine-out evaluation. Then:
    python scripts/build_manifest.py --augment && python scripts/train_model.py
For a LIVE DEMO use --out demo_fakes --prefix demo so these clips are NOT used in training.

NOTE: the network part was not run inside the build sandbox (no access to Microsoft's service);
the conversion/naming logic is unit-tested (tests/test_tools.py).
"""
import argparse
import asyncio
import csv
import sys
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

VOICES = {"aria": "en-US-AriaNeural", "guy": "en-US-GuyNeural",
          "ryan": "en-GB-RyanNeural", "neerja": "en-IN-NeerjaNeural", "prabhat": "en-IN-PrabhatNeural"}
TARGET_SR = 16000


def convert_to_wav16k(src: Path, dst: Path):
    y, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
    sf.write(dst, y, TARGET_SR)


def out_name(prefix: str, voice_key: str, utt_id: str) -> str:
    return f"{prefix}_edgetts_{voice_key}_{utt_id}.wav"


async def synth_one(text, voice, tmp_mp3, retries=3):
    import edge_tts  # imported lazily so the module stays importable/testable without the package
    for attempt in range(retries):
        try:
            await edge_tts.Communicate(text, voice).save(str(tmp_mp3))
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 * (attempt + 1))


async def run(voices, limit, out_dir, prefix):
    path = config.PROCESSED_DIR / "transcripts.csv"
    if not path.exists():
        sys.exit("No transcripts.csv - run scripts/fetch_dev_real_samples.py first.")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["text"].strip()][: limit or None]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp.mp3"
    made = 0
    for r in rows:
        for key in voices:
            await synth_one(r["text"], VOICES[key], tmp)
            convert_to_wav16k(tmp, out_dir / out_name(prefix, key, r["utt_id"]))
            made += 1
    tmp.unlink(missing_ok=True)
    print(f"Generated {made} neural clips -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", default=",".join(["aria", "guy", "ryan", "neerja"]))
    ap.add_argument("--limit", type=int, default=0, help="only the first N transcripts")
    ap.add_argument("--out", default=str(config.SYNTH_DIR))
    ap.add_argument("--prefix", default="synth")
    a = ap.parse_args()
    voices = a.voices.split(",")
    bad = [v for v in voices if v not in VOICES]
    if bad:
        sys.exit(f"Unknown voices {bad}; choose from {sorted(VOICES)}")
    asyncio.run(run(voices, a.limit, Path(a.out), a.prefix))


if __name__ == "__main__":
    main()
