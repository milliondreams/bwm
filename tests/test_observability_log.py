"""Structured event log tests."""
from __future__ import annotations

import json
import os

import pytest


def _read_log(storage) -> list[dict]:
    """Read and parse state/job_log.jsonl from a Storage backend."""
    raw = storage.read_bytes("state/job_log.jsonl").decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class TestEmitEventSchema:
    def test_event_has_required_fields(self, monkeypatch, local_storage):
        from data.observability import log
        # Route log writes to our tmp storage by patching _get_storage
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        log.emit_event("ingest_dera", "financials", "ok_with_records",
                       n_records=13520, wall_seconds=21.9)
        events = _read_log(local_storage)
        assert len(events) == 1
        ev = events[0]
        for k in ("ts", "event_id", "parent_event_id", "pipeline", "modality",
                  "status", "n_records", "wall_seconds"):
            assert k in ev
        assert ev["pipeline"] == "ingest_dera"
        assert ev["status"] == "ok_with_records"
        assert ev["n_records"] == 13520

    def test_each_event_gets_unique_uuid(self, monkeypatch, local_storage):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        e1 = log.emit_event("p", "m", "s")
        e2 = log.emit_event("p", "m", "s")
        assert e1 != e2

    def test_extra_fields_pass_through(self, monkeypatch, local_storage):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        log.emit_event("p", "m", "s", quarter="2024Q1", custom_field="hello")
        events = _read_log(local_storage)
        assert events[0]["quarter"] == "2024Q1"
        assert events[0]["custom_field"] == "hello"

    def test_reserved_keys_not_overwritten(self, monkeypatch, local_storage):
        """Caller-supplied **fields can't overwrite envelope keys (ts, event_id,
        parent_event_id). Named params (pipeline/modality/status) are
        already protected by Python's calling convention; the dedup logic
        guards the remaining envelope.
        """
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        # `ts` is set inside emit_event and must not be overridable via **fields
        log.emit_event("p", "m", "s", **{"ts": "1999-01-01T00:00:00.000000Z"})
        events = _read_log(local_storage)
        # The real ts is something more recent than 1999
        assert not events[0]["ts"].startswith("1999")


class TestStartEndPairing:
    def test_end_event_references_start(self, monkeypatch, local_storage):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        start_id = log.emit_start("ingest_dera", "financials")
        log.emit_end("ingest_dera", "financials", "ok_with_records",
                     start_id, wall_seconds=22.0, n_records=13520)
        events = _read_log(local_storage)
        assert len(events) == 2
        assert events[0]["status"] == "start"
        assert events[1]["parent_event_id"] == events[0]["event_id"]
        assert events[1]["wall_seconds"] == 22.0

    def test_end_event_id_distinct_from_start(self, monkeypatch, local_storage):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        start_id = log.emit_start("p", "m")
        log.emit_end("p", "m", "ok", start_id, wall_seconds=1.0)
        events = _read_log(local_storage)
        assert events[0]["event_id"] != events[1]["event_id"]


class TestStdoutMirroring:
    def test_no_stdout_by_default(self, monkeypatch, local_storage, capsys):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        monkeypatch.delenv("BWM_STRUCTURED_LOG", raising=False)
        log.emit_event("p", "m", "s")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_stdout_when_env_set(self, monkeypatch, local_storage, capsys):
        from data.observability import log
        monkeypatch.setattr(log, "_get_storage", lambda: local_storage)
        monkeypatch.setenv("BWM_STRUCTURED_LOG", "1")
        log.emit_event("p", "m", "s")
        captured = capsys.readouterr()
        # One JSON line, parseable
        line = captured.out.strip()
        assert line
        parsed = json.loads(line)
        assert parsed["pipeline"] == "p"


class TestStorageFailureSwallowed:
    def test_log_failure_does_not_raise(self, monkeypatch, capsys):
        """If the storage backend fails (e.g., read-only mount), emit_event
        must not crash the calling pipeline.
        """
        from data.observability import log

        class FailingStorage:
            def append_bytes(self, path, data):
                raise OSError("read-only filesystem")
        monkeypatch.setattr(log, "_get_storage", lambda: FailingStorage())
        # Should not raise
        log.emit_event("p", "m", "s")
        # Warning lands on stderr
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower()
