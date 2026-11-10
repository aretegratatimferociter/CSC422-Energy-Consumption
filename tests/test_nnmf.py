import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

NNMF_PATH = Path(__file__).resolve().parents[1] / "src" / "NNMF" / "nnmf.py"


def load_nnmf_module():
    spec = importlib.util.spec_from_file_location("nnmf", NNMF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nnmf_converges_on_mw_scale_data():
    hours = np.arange(24)
    profiles = np.vstack([
        np.full(24, 25_000.0),
        4_000 + 7_000 * np.exp(-((hours - 19) ** 2) / 6),
        2_000 + 12_000 * np.exp(-((hours - 16) ** 2) / 18),
    ])
    activations = np.random.default_rng(0).uniform(0.2, 1.2, size=(300, 3))
    X = pd.DataFrame(activations @ profiles, columns=[f"h{h:02d}" for h in hours])

    _, _, _, _, metrics = load_nnmf_module().fit_nnmf(X)
    quality = metrics.set_index("rank")["reconstruction_quality"]

    assert quality.loc[3] > 0.99
    assert (quality.diff().dropna() > -0.005).all()
