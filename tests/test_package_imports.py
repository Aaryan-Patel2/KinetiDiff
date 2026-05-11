"""Canary: verifies the kinetidiff package installs and exposes __version__."""

import importlib


def test_kinetidiff_version() -> None:
    pkg = importlib.import_module("kinetidiff")
    assert pkg.__version__ == "0.1.0"
