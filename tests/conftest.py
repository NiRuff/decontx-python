import os
import tempfile
from pathlib import Path

# Out-of-bounds writes in numba kernels otherwise corrupt memory silently or crash.
# A separate cache dir is needed, because cached kernels compiled without
# bounds checking would otherwise be reused.
os.environ.setdefault("NUMBA_BOUNDSCHECK", "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "decontx_numba_test_cache")
)

import numba  # noqa: E402

numba.config.reload_config()  # numba may already be imported by a pytest plugin
