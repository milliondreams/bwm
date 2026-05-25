"""Verify the run_id helper prefers AZUREML_RUN_ID and falls back to a stable
short uuid for local runs.
"""
from __future__ import annotations

import importlib

import pytest


def _fresh_module():
    """Re-import run_id so the lru_cache is fresh each test."""
    import data.observability.run_id as m
    importlib.reload(m)
    return m


def test_azureml_run_id_wins(monkeypatch):
    monkeypatch.setenv("AZUREML_RUN_ID", "abc123")
    m = _fresh_module()
    assert m.get_run_id() == "abc123"
    assert m.tag("hello") == "[abc123] hello"


def test_local_fallback_is_stable_within_process(monkeypatch):
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    m = _fresh_module()
    rid_first = m.get_run_id()
    rid_second = m.get_run_id()
    assert rid_first == rid_second  # lru_cache stable across calls
    assert rid_first.startswith("local-")
    assert len(rid_first) == len("local-") + 8


def test_tag_formats_correctly(monkeypatch):
    monkeypatch.setenv("AZUREML_RUN_ID", "job-xyz")
    m = _fresh_module()
    assert m.tag("[OK] cik=0000320193") == "[job-xyz] [OK] cik=0000320193"
