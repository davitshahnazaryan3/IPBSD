"""
End-to-end run of the sample case from main.py's __main__ block.

The case is copied to a tmp directory first: IPBSD writes its cache and results
next to the inputs, and sample/sample1 is tracked in git.
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ipbsd.main import Main

REPO_ROOT = Path(__file__).parents[1]
SAMPLE = REPO_ROOT / "sample" / "sample1"

# Written by pandas 1.x and unreadable by pandas >= 2; Hazard regenerates them via HazardFit.
STALE_CACHE = ("coef_hazard.pkl", "fit_hazard.pkl")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path = tmp_path_factory.mktemp("sample1")
    shutil.copytree(SAMPLE, path, dirs_exist_ok=True)
    for name in STALE_CACHE:
        (path / name).unlink(missing_ok=True)
    # Drop the committed results so the run computes them rather than reading them back
    shutil.rmtree(path / "Cache")
    return path


def test_sample_case_runs(case):
    ipbsd = Main(
        case / "ipbsd_input.csv",
        case / "hazard" / "hazard.pkl",
        case / "spo.csv",
        case / "slfoutput",
        limit_eal=1.0,
        target_mafc=2.e-4,
        output_path=case,
        analysis_type=3,
        damping=.05,
        iterate=True,
        maxiter=10,
        fstiff=0.5,
        flag3d=True,
        export=True,
        overstrength=1.0,
        repl_cost=349459.2,
        gravity_cs=None,
        hold_flag=True,
    )

    ipbsd.run_master()

    # Hazard fitting regenerated the cache it found missing
    for name in STALE_CACHE:
        assert (case / name).is_file(), f"{name} was not regenerated"

    # The section-combination phase recomputed the cache that was removed
    assert (case / "Cache").is_dir()

    # Recomputed solutions must still match the reference committed with the sample
    for name in ("solution_cache_space_x.csv", "solution_cache_space_y.csv", "solution_cache_space_gr.csv"):
        reference = pd.read_csv(SAMPLE / "Cache" / name, index_col=0)
        computed = pd.read_csv(case / "Cache" / name, index_col=0)
        assert list(computed.columns) == list(reference.columns)
        numeric = reference.select_dtypes("number")
        assert_allclose(computed[numeric.columns], numeric, rtol=1e-8, err_msg=name)
