"""Verify aml.modality_job._check_env fails fast on missing required secrets
and warns (but does not exit) on missing optional ones.
"""
from __future__ import annotations

import pytest


def _import():
    from aml.modality_job import _check_env, _REQUIRED_ENV, _RECOMMENDED_ENV
    return _check_env, _REQUIRED_ENV, _RECOMMENDED_ENV


def test_macro_requires_fred_api_key(monkeypatch):
    check_env, _, _ = _import()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        check_env("macro")
    assert "FRED_API_KEY" in str(exc.value)


def test_macro_passes_when_key_set(monkeypatch):
    check_env, _, _ = _import()
    monkeypatch.setenv("FRED_API_KEY", "k")
    check_env("macro")  # should not raise


def test_hiring_warns_but_does_not_raise_without_key(monkeypatch, capsys):
    check_env, _, _ = _import()
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    check_env("hiring")  # must not raise
    out = capsys.readouterr().out
    assert "BLS_API_KEY" in out and "warn" in out.lower()


def test_market_has_no_required_or_recommended(monkeypatch):
    check_env, _, _ = _import()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    check_env("market")  # no required, no soft warning
