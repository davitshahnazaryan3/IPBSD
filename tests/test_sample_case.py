"""
End-to-end run of the sample case from main.py's __main__ block.

The case is copied to a tmp directory first: IPBSD writes its cache and results
next to the inputs, and sample/sample1 is tracked in git.

The expected values were recorded on pandas 3.0.6 / numpy 2.5.3 / openseespy 3.8.0.0 and
agree with the reference cache committed with the sample to within 5e-13. They are rounded
and asserted at rtol=1e-3, which catches drift from refactoring src/ without tripping on
solver noise.
"""
import pickle
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

SOLUTION_CACHES = ("solution_cache_space_x.csv", "solution_cache_space_y.csv", "solution_cache_space_gr.csv")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """Runs the sample case once from a clean copy and yields its output directory."""
    path = tmp_path_factory.mktemp("sample1")
    shutil.copytree(SAMPLE, path, dirs_exist_ok=True)
    for name in STALE_CACHE:
        (path / name).unlink(missing_ok=True)
    # Drop the committed results so the run computes them rather than reading them back
    shutil.rmtree(path / "Cache")

    Main(
        path / "ipbsd_input.csv",
        path / "hazard" / "hazard.pkl",
        path / "spo.csv",
        path / "slfoutput",
        limit_eal=1.0,
        target_mafc=2.e-4,
        output_path=path,
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
    ).run_master()

    return path


def read_cache(case, name):
    with open(case / "Cache" / name, "rb") as file:
        return pickle.load(file)


def test_run_exports_expected_artefacts(case):
    # Hazard fitting regenerated the cache it found missing
    for name in STALE_CACHE:
        assert (case / name).is_file(), f"{name} was not regenerated"

    assert (case / "Cache").is_dir()
    for name in SOLUTION_CACHES:
        assert (case / "Cache" / name).is_file(), f"{name} was not exported"


def test_hazard_fit_coefficients(case):
    coefs = read_cache(case, "input_cache.pickle")["coefs"]["PGA"]
    assert_allclose(coefs, [6.167e-4, 2.111, 0.1763], rtol=1e-3)


def test_expected_annual_loss_and_limit_states(case):
    loss = read_cache(case, "lossCurve.pickle")

    assert loss["PLS"] == ["OLS", "SLS", "CLS"]
    assert_allclose(loss["eal"], 0.5977, rtol=1e-3)
    assert_allclose(loss["mafe"], [0.10026, 0.01726, 2.e-4], rtol=1e-3)
    assert_allclose(loss["y"], [0.01, 0.08373, 1.0], rtol=1e-3)

    # EAL must land under the 1.0% limit passed to Main
    assert loss["eal"] < 1.0


def test_sls_design_spectrum(case):
    spectrum = pd.read_csv(case / "Cache" / "sls_spectrum.csv", index_col=0)

    assert_allclose(spectrum.loc[0.2, "Sa"], 0.3601, rtol=1e-3)
    assert_allclose(spectrum.loc[0.2, "Sd"], 0.358, rtol=1e-3)


def test_feasible_solutions(case):
    elements = pd.read_csv(case / "Cache" / "elements_space.csv", index_col=0)
    solutions = pd.read_csv(case / "Cache" / "solution_cache_space_x.csv", index_col=0)

    # 240 combinations satisfy the period bounds, 126 of them survive modal verification
    assert len(elements) == 240
    assert len(solutions) == 126
    assert list(solutions.columns) == ["he1", "hi1", "b1", "h1", "he2", "hi2", "b2", "h2",
                                       "T", "Weight", "Mstar", "Part Factor"]

    assert_allclose([solutions["T"].min(), solutions["T"].max()],
                    [0.199, 0.3685], rtol=1e-3)
    assert_allclose([solutions["Weight"].min(), solutions["Weight"].max()],
                    [2002.75, 3785.5], rtol=1e-3)


def test_solutions_match_committed_reference(case):
    """Guards against drift away from the results the sample was originally generated with."""
    for name in SOLUTION_CACHES:
        reference = pd.read_csv(SAMPLE / "Cache" / name, index_col=0)
        computed = pd.read_csv(case / "Cache" / name, index_col=0)

        assert list(computed.columns) == list(reference.columns)
        numeric = reference.select_dtypes("number")
        assert_allclose(computed[numeric.columns], numeric, rtol=1e-3, err_msg=name)
