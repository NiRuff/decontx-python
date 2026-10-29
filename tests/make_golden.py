"""Regenerate the golden reference. Only run after an intended numerical change."""
from pathlib import Path

import numpy as np
from anndata import AnnData

from decontx import decontx
from _sim import simulate

X, _, z, _ = simulate()
a = AnnData(X.astype(np.float32))
a.obs["leiden"] = z.astype(str)
decontx(a, verbose=False)
out = Path(__file__).parent / "golden" / "sim_contamination.npy"
np.save(out, a.obs["decontX_contamination"].values)
print(f"wrote {out}")
