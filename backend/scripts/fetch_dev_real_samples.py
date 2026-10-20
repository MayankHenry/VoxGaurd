"""
Fetch a small set of REAL human speech clips for development.

Source: LJSpeech clips bundled in the public coqui-ai/TTS test-data folder on GitHub
(LJSpeech is public domain). Works on any machine with git + internet.

IMPORTANT: this is a *development* corpus: 32 clips, ONE speaker, studio quality.
It is enough to validate the pipeline end-to-end but NOT enough to claim real-world
accuracy. For the final evaluation add ASVspoof2019 bonafide clips (see
extract_asvspoof_subset.py) and your own recordings (prepare_real_samples.py).

Usage:  python scripts/fetch_dev_real_samples.py
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from scripts.prepare_real_samples import validate_and_convert
import soundfile as sf


# The public repo only ships transcripts for 8 of the 32 clips. For the rest we speak a
# similar-style sentence instead (flagged matched=False in transcripts.csv). Text is
# therefore matched for 8/32 utterances only - acoustic features are text-weak, but this
# is stated openly in the model card.
FALLBACK_SENTENCES = [
    "The committee reviewed the annual report and requested several changes before approval.",
    "Early in the morning the harbour was quiet, and only a few boats had left the shore.",
    "Although the evidence was incomplete, the investigators continued to examine every record.",
    "Visitors to the old library often remark on the height of its painted ceilings.",
    "The new regulations will take effect at the beginning of the following financial year.",
    "She walked slowly along the river, thinking carefully about what she would say.",
    "Several engineers argued that the bridge should be repaired rather than replaced.",
    "In the evening the streets filled with people returning from the market.",
    "The manuscript was copied by hand and preserved in a small monastery for centuries.",
    "Prices rose steadily throughout the autumn, causing concern among local merchants.",
    "He explained the procedure in detail, pausing frequently to answer questions.",
    "The expedition set out at dawn and reached the mountain pass by noon.",
    "A careful examination of the document revealed several minor inconsistencies.",
    "Many of the older buildings in the district were demolished during the reconstruction.",
    "The speaker thanked the audience and promised to return the following year.",
    "Rainfall was unusually heavy that season, and the crops suffered considerably.",
    "The museum acquired a remarkable collection of maps from a private owner.",
    "They agreed to meet again after the committee had published its findings.",
    "The school was founded by a group of teachers who believed in practical education.",
    "Telephone lines were installed across the valley during the last decade.",
    "A long discussion followed, but no final decision was reached that afternoon.",
    "The garden, which had been neglected for years, was restored by volunteers.",
    "Witnesses described the events in remarkably similar terms.",
    "The factory employed nearly two hundred workers before it closed its doors.",
]

REPO = "https://github.com/coqui-ai/TTS"
SUBDIR = "tests/data/ljspeech"
SPEAKER = "ljspeech"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ljs_"))
    try:
        print("Sparse-cloning LJSpeech sample clips ...")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(tmp / "repo")],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp / "repo"), "sparse-checkout", "set", SUBDIR],
                       check=True, capture_output=True)
        src = tmp / "repo" / SUBDIR

        transcripts = {}
        with open(src / "metadata.csv", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                transcripts[row[0]] = row[1]

        for old in config.REAL_DIR.glob(f"real_{SPEAKER}_*.wav"):
            old.unlink()

        kept, rejected, utts = 0, 0, []
        for wav in sorted((src / "wavs").glob("*.wav")):
            try:
                y, sr = validate_and_convert(wav)
            except ValueError as e:
                print(f"SKIPPED {wav.name}: {e}")
                rejected += 1
                continue
            sf.write(config.REAL_DIR / f"real_{SPEAKER}_{wav.stem}.wav", y, sr)
            utts.append(wav.stem)
            kept += 1

        out = config.PROCESSED_DIR / "transcripts.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "text", "matched"])
            fb = iter(FALLBACK_SENTENCES)
            for u in utts:
                if transcripts.get(u):
                    w.writerow([u, transcripts[u], True])
                else:
                    w.writerow([u, next(fb), False])
        print(f"Real clips saved: {kept} (rejected {rejected}) -> {config.REAL_DIR}")
        print(f"Transcripts: {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
