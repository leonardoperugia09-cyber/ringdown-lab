"""Keep the interactive app (app/) consistent with the Python code."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from ringdown_lab.qnm import COEFFS, MSUN_SECONDS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PHYSICS = (APP / "physics.js").read_text(encoding="utf-8")
PAGE = (APP / "index.html").read_text(encoding="utf-8")


def test_page_loads_physics_with_a_relative_path():
    """GitHub Pages serves the site under /overtone-explorer/, so no absolute paths."""
    assert '<script src="physics.js"></script>' in PAGE
    assert 'src="/' not in PAGE and 'href="/' not in PAGE


def test_single_file_build_is_up_to_date():
    spec = importlib.util.spec_from_file_location("build_single", APP / "build_single.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = (APP / "overtone_explorer.html").read_text(encoding="utf-8")
    assert built == mod.build(), "run: python app/build_single.py"
    assert PHYSICS in built


@pytest.mark.parametrize("mode", ["220", "221"])
def test_app_uses_the_same_qnm_coefficients(mode):
    found = re.search(rf'"{mode}": \[([^\]]+)\]', PHYSICS).group(1)
    assert tuple(float(x) for x in found.split(",")) == COEFFS[mode]


def test_app_constants():
    msun = float(re.search(r"MSUN_S = ([\d.e+-]+);", PHYSICS).group(1))
    assert msun == MSUN_SECONDS
    assert "FS = 4096" in PHYSICS


def test_app_threshold_matches_experiment_c():
    results = ROOT / "results" / "detection.json"
    if not results.exists():
        pytest.skip("run python run_detect.py first")
    thr = json.loads(results.read_text())["experiment_c"]["threshold"]
    app_thr = float(re.search(r"DCHI2_THRESHOLD = ([\d.]+);", PHYSICS).group(1))
    assert app_thr == pytest.approx(thr, abs=0.01)


def test_page_has_no_host_specific_tags():
    """The site must work on any host, so no viewer-specific meta tags."""
    for f in ("index.html", "physics.js"):
        assert "viewport-fit-sides" not in (APP / f).read_text(encoding="utf-8")
