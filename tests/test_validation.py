"""Validation tests: ground-truth recovery, batch edge cases, regression, threading."""
from pathlib import Path

import numba
import numpy as np
import pytest
from anndata import AnnData

from decontx import decontx
from decontx.model import DecontXModel

from _sim import simulate

GOLDEN = Path(__file__).parent / "golden" / "sim_contamination.npy"


@pytest.fixture(scope="module")
def sim():
    return simulate()


def _adata(X, z):
    a = AnnData(X.astype(np.float32))
    a.obs["leiden"] = z.astype(str)
    return a


def test_recovers_ground_truth(sim):
    X, native, z, true_c = sim
    a = _adata(X, z)
    decontx(a, verbose=False)
    est = a.obs["decontX_contamination"].values
    corrected = a.layers["decontX_counts"]

    assert np.corrcoef(est, true_c)[0, 1] > 0.95
    assert abs(est.mean() - true_c.mean()) < 0.03
    # most ambient counts removed, most native counts kept
    assert (X - corrected).sum() / (X - native).sum() > 0.8
    assert corrected.multiply(native > 0).sum() / native.sum() > 0.9


@pytest.mark.parametrize("missing", [0, 2])
def test_batch_with_missing_cluster(sim, missing):
    """A batch lacking a cluster (low or middle id) must not break indexing."""
    X, _, z, true_c = sim
    a = _adata(X, z)
    rng = np.random.default_rng(0)
    batch = np.where(z == missing, "A", rng.choice(["A", "B"], len(z)))
    a.obs["batch"] = batch
    decontx(a, batch_key="batch", verbose=False)
    est = a.obs["decontX_contamination"].values

    assert np.all((est >= 0) & (est <= 1))
    for b in ("A", "B"):
        m = batch == b
        assert np.corrcoef(est[m], true_c[m])[0, 1] > 0.9


def test_loglik_recorded_without_verbose(sim):
    X, _, z, _ = sim
    res = DecontXModel(verbose=False).fit_transform(X, z + 1)
    assert len(res["log_likelihood"]) > 0


def test_regression_against_golden(sim):
    """Guards against unintended numerical changes. If a change is intended,
    regenerate with `python tests/make_golden.py` and document it in the changelog."""
    X, _, z, _ = sim
    a = _adata(X, z)
    decontx(a, verbose=False)
    diff = np.abs(a.obs["decontX_contamination"].values - np.load(GOLDEN))
    # Tight enough to catch the 0.2 -> 0.3 changes (mean 5e-4 and 9e-4), loose enough
    # for platform/thread rounding that can shift convergence by one iteration.
    assert diff.max() < 2e-3 and diff.mean() < 2e-4


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2, reason="needs >= 2 threads")
def test_thread_count_invariance(sim):
    X, _, z, _ = sim
    out = []
    for n in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
        numba.set_num_threads(n)
        a = _adata(X, z)
        decontx(a, verbose=False)
        out.append(a.obs["decontX_contamination"].values)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    np.testing.assert_allclose(out[0], out[1], atol=1e-3)
