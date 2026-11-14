# VoxGuard - experiment log (honest results)

All numbers come from `python scripts/train_model.py` on the development corpus
(47 real clips from ~6 sources, 256 synthetic clips from 3 classical TTS engines; every clip also
in two telephone-channel variants). Test folds always contain clean AND degraded rows.

## 1. The 100% that was not real
First version: grouped CV on LJSpeech-vs-TTS scored ~100 %. The same model then called **14 of 15
unseen real clips "synthetic"**: it had learned the speaker/recording channel, not "human vs AI".
-> Evaluation was rebuilt around leave-speakers-out (LOSO) and leave-one-engine-out (LOEO), with
speaker/engine-balanced sample weights. In-distribution CV is now reported as "optimistic" only.

## 2. Does telephone-channel augmentation help? (clean-only vs augmented training)

| | Clean-only training | Augmented training (shipped) |
|---|---|---|
| Selection score (0.5 LOSO + 0.5 LOEO) | 0.753 (xgboost, no_mfcc_mean) | **0.777** (xgboost, all) |
| Unseen espeak - synthetic detected / EER | 91 % / 0.084 | **100 % / 0.033** |
| Unseen festival - synthetic detected / EER | 59 % / 0.141 | **92 % / 0.072** |
| Unseen flite - synthetic detected / EER | 40 % / 0.263 | **54 % / 0.254** |
| Unseen-engine detection on phone-degraded audio (espeak / festival / flite) | 86 / 52 / 36 % | **100 / 88 / 56 %** |
| Unseen REAL speakers recognised as real (macro) | **0.66** | 0.37 |

Reading: augmentation clearly improves robustness to phone-quality audio and to unseen TTS engines,
but **recognition of unseen real speakers got worse** (more real people flagged as AI). Only ~6 real
sources exist, so speaker generalisation is the limiting factor, not the algorithm. The augmented model
is shipped because the product targets phone calls, and the frontend shows an "uncertain" verdict
(max probability < 0.65) instead of forcing a call.

## 3. What would fix it (next data, not next algorithm)
1. `scripts/import_asvspoof.py` - ASVspoof2019 LA: many real speakers + attacks A01-A06 (train) and A07-A19 (hold-out).
2. `scripts/generate_neural_samples.py` - neural TTS voices as extra generator families.
3. `scripts/evaluate_external.py` - final test on the untouched hold-out, reporting EER.
