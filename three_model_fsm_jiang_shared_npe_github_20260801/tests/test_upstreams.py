import subprocess

from khoo_vs_jiang.upstreams import (
    EXPECTED_JIANG_COMMIT,
    EXPECTED_KHOO_SHA256,
    load_upstreams,
)


def test_upstreams_resolve_to_external_repositories() -> None:
    modules = load_upstreams()
    assert modules.jiang_root.name == "Structured_Score_Matching"
    assert modules.khoo_root.name == "model3_upstream"
    assert EXPECTED_JIANG_COMMIT
    assert len(EXPECTED_KHOO_SHA256) == 3
    assert modules.jiang_utils.__file__.startswith(str(modules.jiang_root))
    assert modules.khoo_oneparam.__file__.startswith(str(modules.khoo_root))


def test_upstreams_have_no_tracked_modifications() -> None:
    modules = load_upstreams()
    for root in (modules.jiang_root, modules.khoo_root):
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            # load_upstreams has already verified the frozen SHA-256 values
            # for a deployed Khoo source directory without Git metadata.
            assert root == modules.khoo_root
            continue
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=no"],
            text=True,
        )
        assert status == ""
