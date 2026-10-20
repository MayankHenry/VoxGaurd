import numpy as np
import pandas as pd
from scripts.train_model import validate_dataset, sample_weights


def _df(real, synth, groups=10):
    rows = [{"label": "real", "family": "human", "variant": f"spk{i % 2}", "group": f"g{i % groups}"} for i in range(real)]
    rows += [{"label": "synthetic", "family": "espeak" if i % 2 else "flite", "variant": "a", "group": f"g{i % groups}"} for i in range(synth)]
    return pd.DataFrame(rows)


def test_guard_refuses_single_class():
    assert not validate_dataset(_df(0, 20))


def test_guard_refuses_too_few_per_class():
    assert not validate_dataset(_df(3, 20))


def test_guard_accepts_reasonable_dataset():
    assert validate_dataset(_df(10, 20))


def test_sample_weights_balance_classes_and_sources():
    df = _df(40, 10)
    w = sample_weights(df)
    real_w = w[df.label == "real"].sum()
    synth_w = w[df.label == "synthetic"].sum()
    assert np.isclose(real_w, synth_w)
    # each real speaker carries equal total weight
    per_spk = [w[(df.label == "real") & (df.variant == s)].sum() for s in ("spk0", "spk1")]
    assert np.isclose(per_spk[0], per_spk[1])
