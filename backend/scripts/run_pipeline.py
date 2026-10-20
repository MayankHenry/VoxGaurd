"""
One command to rebuild everything: fetch real clips -> generate synthetic -> manifest -> train.

    python scripts/run_pipeline.py              # full run (needs internet + git + TTS engines)
    python scripts/run_pipeline.py --train-only # reuse committed manifest.csv (no audio / TTS needed)

TTS engines (Linux/WSL/macOS): espeak-ng, flite, festival (text2wave).
  sudo apt install espeak-ng flite festival festvox-kallpc16k festvox-ellpc11k
On plain Windows use WSL for the data-generation steps, or --train-only.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("Fetch real clips (LJSpeech)", "fetch_dev_real_samples.py"),
    ("Fetch extra real speakers", "fetch_extra_real_speakers.py"),
    ("Generate synthetic clips", "generate_synthetic_samples.py"),
    ("Build feature manifest", "build_manifest.py"),
    ("Train + evaluate", "train_model.py"),
]


def main():
    train_only = "--train-only" in sys.argv
    if not train_only:
        missing = [t for t in ("git", "espeak-ng", "flite", "text2wave") if shutil.which(t) is None]
        if missing:
            print(f"Missing tools: {missing}\nInstall them (see docstring) or use --train-only.")
            sys.exit(1)
    for title, script in (STEPS[-1:] if train_only else STEPS):
        print(f"\n=== {title} ===")
        if subprocess.run([sys.executable, str(HERE / script)]).returncode != 0:
            print(f"Step failed: {script}")
            sys.exit(1)
    print("\nDone. Start the API:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
