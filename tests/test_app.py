"""Keep the interactive app (app/) consistent with the Python code."""

import json
import re
from pathlib import Path

import pytest

from ringdown_lab.qnm import COEFFS, MSUN_SECONDS

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app" / "overtone_explorer.html").read_text(encoding="utf-8")


def test_standalone_copy_is_in_sync():
    """app/index.html = the published page plus a doctype, for opening from disk."""
    standalone = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    assert standalone == "<!doctype html>\n" + APP


@pytest.mark.parametrize("mode", ["220", "221"])
def test_app_uses_the_same_qnm_coefficients(mode):
    found = re.search(rf'"{mode}": \[([^\]]+)\]', APP).group(1)
    assert tuple(float(x) for x in found.split(",")) == COEFFS[mode]


def test_app_constants():
    msun = float(re.search(r"MSUN_S = ([\d.e+-]+);", APP).group(1))
    assert msun == MSUN_SECONDS
    assert "FS = 4096" in APP


def test_app_threshold_matches_experiment_c():
    results = ROOT / "results" / "detection.json"
    if not results.exists():
        pytest.skip("run python run_detect.py first")
    thr = json.loads(results.read_text())["experiment_c"]["threshold"]
    app_thr = float(re.search(r"DCHI2_THRESHOLD = ([\d.]+);", APP).group(1))
    assert app_thr == pytest.approx(thr, abs=0.01)
