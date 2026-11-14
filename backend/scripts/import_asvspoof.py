"""
Import a balanced subset of ASVspoof2019 LA into VoxGuard's data folders.

Get the data: https://datashare.ed.ac.uk/handle/10283/3336  (download LA.zip, ~7 GB, and extract).
Partitions: *train* contains attacks A01-A06, *eval* contains A07-A19 (attacks unseen in training),
so:  --role train   -> data/raw/real + data/raw/synthetic   (used by build_manifest / train_model)
     --role holdout -> data/holdout/real + data/holdout/fake (used ONLY by evaluate_external.py)

    python scripts/import_asvspoof.py --role train \
        --protocol LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --audio    LA/ASVspoof2019_LA_train/flac --n-bonafide 150 --n-spoof 300

    python scripts/import_asvspoof.py --role holdout \
        --protocol LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
        --audio    LA/ASVspoof2019_LA_eval/flac --n-bonafide 100 --n-spoof 200

Protocol line:  <speaker> <file_id> - <attack|-> <bonafide|spoof>
Files are written with VoxGuard's naming convention so the speaker (real) and attack id
(synthetic family) are recovered automatically for leave-speaker-out / leave-attack-out
evaluation:  real_<speaker>_<file_id>.wav   synth_asv<ATTACK>_<ATTACK>_<file_id>.wav
Clips are capped per speaker / per attack so no single speaker or attack dominates.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

TARGET_SR = 16000


def parse_protocol(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        speaker, file_id, attack, label = parts[0], parts[1], parts[3], parts[4]
        if label in ("bonafide", "spoof"):
            rows.append({"speaker": speaker, "file_id": file_id, "attack": attack, "label": label})
    return rows


def balanced_pick(items, key, n, rng):
    """Round-robin over `key` groups so every speaker / attack contributes about equally."""
    buckets = defaultdict(list)
    for it in items:
        buckets[it[key]].append(it)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    picked = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop())
    return picked


def _convert(src: Path, dst: Path):
    y, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
    sf.write(dst, y, TARGET_SR)


def import_asvspoof(protocol, audio_dir, out_real, out_fake, n_bonafide, n_spoof, seed=42):
    rng = random.Random(seed)
    rows = parse_protocol(protocol)
    bona = balanced_pick([r for r in rows if r["label"] == "bonafide"], "speaker", n_bonafide, rng)
    spoof = balanced_pick([r for r in rows if r["label"] == "spoof"], "attack", n_spoof, rng)
    out_real, out_fake, audio_dir = Path(out_real), Path(out_fake), Path(audio_dir)
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)
    made_r = made_f = missing = 0
    for r in bona:
        src = audio_dir / f"{r['file_id']}.flac"
        if not src.exists():
            missing += 1
            continue
        _convert(src, out_real / f"real_{r['speaker'].replace('_', '')}_{r['file_id']}.wav")
        made_r += 1
    for r in spoof:
        src = audio_dir / f"{r['file_id']}.flac"
        if not src.exists():
            missing += 1
            continue
        _convert(src, out_fake / f"synth_asv{r['attack']}_{r['attack']}_{r['file_id']}.wav")
        made_f += 1
    return {"real": made_r, "spoof": made_f, "missing": missing,
            "speakers": len({r["speaker"] for r in bona}), "attacks": sorted({r["attack"] for r in spoof})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--audio", required=True, help="folder with the .flac files")
    ap.add_argument("--role", choices=["train", "holdout"], default="train")
    ap.add_argument("--n-bonafide", type=int, default=150)
    ap.add_argument("--n-spoof", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.role == "train":
        out_real, out_fake = config.REAL_DIR, config.SYNTH_DIR
    else:
        out_real, out_fake = config.DATA_DIR / "holdout" / "real", config.DATA_DIR / "holdout" / "fake"
    res = import_asvspoof(a.protocol, a.audio, out_real, out_fake, a.n_bonafide, a.n_spoof, a.seed)
    print(f"Imported {res['real']} bonafide ({res['speakers']} speakers) and {res['spoof']} spoof clips "
          f"(attacks {', '.join(res['attacks'])}); {res['missing']} listed files not found.")
    print(f"-> {out_real}\n-> {out_fake}")
    if a.role == "train":
        print("Next: python scripts/build_manifest.py --augment && python scripts/train_model.py")
    else:
        print(f"Next: python scripts/evaluate_external.py --real {out_real} --fake {out_fake} --variants none,codec,noisy")


if __name__ == "__main__":
    main()
