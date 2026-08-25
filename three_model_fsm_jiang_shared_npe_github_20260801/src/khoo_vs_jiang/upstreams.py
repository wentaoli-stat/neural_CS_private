"""Read-only imports of the two upstream implementations.

Nothing from either upstream is copied into this repository.  The path checks
also make accidental imports from a similarly named local module fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


EXPECTED_JIANG_COMMIT = "fb273f0e1bbfca2d1d752c97d1f9d431dd1039c9"
EXPECTED_KHOO_SHA256 = {
    "run_hmm_oneparam_local_ncs_gru_experiment.py": (
        "c7f0a98ee6c32658862ec9fa6cce50f58a6d90c20a90172f318c5a9ec5ec13a1"
    ),
    "run_hmm_common_factor_amortized_fsm_experiment.py": (
        "4f0154e23c0dbe0c223010600955c04e2c13d7e8dae7dd2db2b550eb1a5575c5"
    ),
    "run_hmm_common_factor_fsm_experiment.py": (
        "c2be4bdc753a1dbc382ddb57591dd3b4020a526d4e3186c359ab8c0a1721a5e8"
    ),
}


@dataclass(frozen=True)
class UpstreamModules:
    jiang_root: Path
    khoo_root: Path
    jiang_utils: ModuleType
    khoo_amortized: ModuleType
    khoo_fixed: ModuleType
    khoo_oneparam: ModuleType


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root(env_name: str, fallback: Path, sentinel: Path) -> Path:
    value = os.environ.get(env_name)
    root = Path(value).expanduser().resolve() if value else fallback.resolve()
    if not (root / sentinel).is_file():
        raise FileNotFoundError(
            f"{env_name}={root} does not contain required file {sentinel}"
        )
    return root


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _assert_from(module: ModuleType, root: Path) -> None:
    module_file = Path(str(module.__file__)).resolve()
    if not module_file.is_relative_to(root):
        raise ImportError(f"{module.__name__} came from {module_file}, not {root}")


def load_upstreams(
    *,
    require_frozen_jiang_commit: bool = True,
    require_frozen_khoo_files: bool = True,
) -> UpstreamModules:
    workspace = _workspace_root()
    jiang_root = _resolve_root(
        "JIANG_SSM_REPO",
        workspace / "external" / "Structured_Score_Matching",
        Path("MLE/utils_sm.py"),
    )
    khoo_root = _resolve_root(
        "KHOO_FSM_REPO",
        workspace / "experiments" / "model3_upstream",
        Path("run_hmm_oneparam_local_ncs_gru_experiment.py"),
    )

    jiang_head = _git_head(jiang_root)
    if (
        require_frozen_jiang_commit
        and jiang_head is not None
        and jiang_head != EXPECTED_JIANG_COMMIT
    ):
        raise RuntimeError(
            "Jiang upstream commit changed: "
            f"expected {EXPECTED_JIANG_COMMIT}, found {jiang_head}"
        )
    if require_frozen_khoo_files:
        for relative, expected in EXPECTED_KHOO_SHA256.items():
            path = khoo_root / relative
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(
                    "Khoo upstream file changed: "
                    f"{path} expected sha256 {expected}, found {actual}"
                )

    _prepend(jiang_root)
    _prepend(khoo_root)
    jiang_utils = importlib.import_module("MLE.utils_sm")
    khoo_amortized = importlib.import_module(
        "run_hmm_common_factor_amortized_fsm_experiment"
    )
    khoo_fixed = importlib.import_module("run_hmm_common_factor_fsm_experiment")
    khoo_oneparam = importlib.import_module(
        "run_hmm_oneparam_local_ncs_gru_experiment"
    )

    _assert_from(jiang_utils, jiang_root)
    _assert_from(khoo_amortized, khoo_root)
    _assert_from(khoo_fixed, khoo_root)
    _assert_from(khoo_oneparam, khoo_root)
    return UpstreamModules(
        jiang_root=jiang_root,
        khoo_root=khoo_root,
        jiang_utils=jiang_utils,
        khoo_amortized=khoo_amortized,
        khoo_fixed=khoo_fixed,
        khoo_oneparam=khoo_oneparam,
    )
