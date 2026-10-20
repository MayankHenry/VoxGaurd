"""
Generate the SYNTHETIC class with two independent offline TTS engines.

Each real clip's transcript is re-spoken by every engine, so real and synthetic
classes contain the SAME sentences (text content can't act as a shortcut) and the
engines act as different "generators" for leave-one-engine-out evaluation.

  family "espeak": espeak-ng  (formant synthesis)         - 2 voice variants
  family "flite" : CMU Flite  (concatenative/HTS synth)   - 4 voices
  family "festival": Festival diphone synthesis            - 2 voices (kal, el)

Honest limitation: these are classical TTS engines, not neural voice cloning.
Add ElevenLabs/Coqui/Bark/OpenVoice outputs (generated on your machine) to the
same folder, named synth_<family>_<variant>_<uttid>.wav, to extend the benchmark.

Usage:  python scripts/generate_synthetic_samples.py
Requires: espeak-ng, flite, festival  (apt install espeak-ng flite festival festvox-kallpc16k festvox-ellpc11k)
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import librosa
import soundfile as sf
from app import config

ESPEAK_VARIANTS = {"a": ["-v", "en-us", "-p", "50", "-s", "170"],
                   "b": ["-v", "en-gb", "-p", "35", "-s", "150"]}
FLITE_VOICES = ["awb", "kal16", "rms", "slt"]
FESTIVAL_VOICES = {"kal": "(voice_kal_diphone)", "el": "(voice_el_diphone)"}
TARGET_SR = 16000


def load_transcripts():
    path = config.PROCESSED_DIR / "transcripts.csv"
    if not path.exists():
        print("No transcripts.csv - run scripts/fetch_dev_real_samples.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return [(r["utt_id"], r["text"]) for r in csv.DictReader(f) if r["text"].strip()]


def save_standardised(tmp_wav: Path, out_path: Path):
    y, _ = librosa.load(str(tmp_wav), sr=TARGET_SR, mono=True)  # same 16 kHz mono as real clips
    sf.write(out_path, y, TARGET_SR)


def main():
    items = load_transcripts()
    for old in config.SYNTH_DIR.glob("synth_*.wav"):
        old.unlink()
    made, failed = 0, 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "t.wav"
        for utt, text in items:
            jobs = []
            for var, args in ESPEAK_VARIANTS.items():
                jobs.append((f"espeak_{var}_{utt}", ["espeak-ng", *args, "-w", str(tmp), text]))
            for voice in FLITE_VOICES:
                jobs.append((f"flite_{voice}_{utt}", ["flite", "-voice", voice, "-t", text, "-o", str(tmp)]))
            txt = Path(td) / "t.txt"
            txt.write_text(text, encoding="utf-8")
            for vname, scheme in FESTIVAL_VOICES.items():
                jobs.append((f"festival_{vname}_{utt}", ["text2wave", "-eval", scheme, str(txt), "-o", str(tmp)]))
            for name, cmd in jobs:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0 or not tmp.exists():
                    failed += 1
                    print(f"FAILED {name}: {r.stderr.strip()[:80]}")
                    continue
                save_standardised(tmp, config.SYNTH_DIR / f"synth_{name}.wav")
                tmp.unlink()
                made += 1
    print(f"Synthetic clips generated: {made} (failed {failed}) -> {config.SYNTH_DIR}")


if __name__ == "__main__":
    main()
