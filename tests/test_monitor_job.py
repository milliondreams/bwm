"""Monitor CLI tests.

We can't exercise the real `az` CLI in CI; instead we mock the
`_poll_status` function to return scripted status sequences and verify
the event emission + exit code behavior.
"""
from __future__ import annotations

import io
import json

import pytest


def _capture_events(capsys) -> list[dict]:
    """Read all JSON lines emitted on stdout in this test."""
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


class TestMonitor:
    def test_emits_start_then_terminal(self, monkeypatch, capsys):
        """Job starts Running, completes; expect start + transition + end events."""
        from aml.cli import monitor_job

        # Scripted status sequence: Running → Running → Completed
        sequence = iter(["Running", "Running", "Completed"])
        monkeypatch.setattr(monitor_job, "_poll_status",
                            lambda *a, **kw: next(sequence))
        monkeypatch.setattr(monitor_job.time, "sleep", lambda s: None)

        rc = monitor_job.monitor("test_job", "rg", "ws", poll_seconds=0)
        assert rc == 0
        events = _capture_events(capsys)
        kinds = [e["event"] for e in events]
        assert "monitor_start" in kinds
        assert "status_transition" in kinds
        assert "monitor_end" in kinds
        end_event = [e for e in events if e["event"] == "monitor_end"][0]
        assert end_event["final_status"] == "Completed"

    def test_failed_terminal_exits_nonzero(self, monkeypatch, capsys):
        from aml.cli import monitor_job
        monkeypatch.setattr(monitor_job, "_poll_status",
                            lambda *a, **kw: "Failed")
        monkeypatch.setattr(monitor_job.time, "sleep", lambda s: None)
        rc = monitor_job.monitor("test_job", "rg", "ws", poll_seconds=0)
        assert rc == 1
        events = _capture_events(capsys)
        end = [e for e in events if e["event"] == "monitor_end"][0]
        assert end["final_status"] == "Failed"

    def test_one_event_per_transition(self, monkeypatch, capsys):
        """Repeated identical statuses should NOT generate spam events."""
        from aml.cli import monitor_job
        # 5 polls all "Running", then "Completed"
        sequence = iter(["Running"] * 5 + ["Completed"])
        monkeypatch.setattr(monitor_job, "_poll_status",
                            lambda *a, **kw: next(sequence))
        monkeypatch.setattr(monitor_job.time, "sleep", lambda s: None)
        rc = monitor_job.monitor("test_job", "rg", "ws", poll_seconds=0)
        assert rc == 0
        events = _capture_events(capsys)
        transitions = [e for e in events if e["event"] == "status_transition"]
        # Two transitions: None→Running, Running→Completed
        assert len(transitions) == 2
        assert transitions[0]["to"] == "Running"
        assert transitions[1]["from"] == "Running"
        assert transitions[1]["to"] == "Completed"

    def test_az_missing_returns_error(self, monkeypatch, capsys):
        from aml.cli import monitor_job
        monkeypatch.setattr(monitor_job, "_poll_status",
                            lambda *a, **kw: "AzCliMissing")
        rc = monitor_job.monitor("test_job", "rg", "ws", poll_seconds=0)
        assert rc == 1
        events = _capture_events(capsys)
        assert any(e.get("event") == "error" for e in events)

    def test_max_wait_timeout(self, monkeypatch, capsys):
        from aml.cli import monitor_job
        # Always Running; max_wait_seconds forces exit
        monkeypatch.setattr(monitor_job, "_poll_status",
                            lambda *a, **kw: "Running")
        # Simulate elapsed time by patching time.monotonic
        ticks = iter([0.0, 0.1, 0.2, 1000.0, 1001.0])  # last value crosses max_wait
        monkeypatch.setattr(monitor_job.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(monitor_job.time, "sleep", lambda s: None)
        rc = monitor_job.monitor("test_job", "rg", "ws",
                                 poll_seconds=0, max_wait_seconds=100)
        assert rc == 1
        events = _capture_events(capsys)
        assert any(e.get("event") == "monitor_timeout" for e in events)


class TestPollStatus:
    def test_invokes_az_with_correct_args(self, monkeypatch):
        """_poll_status should call az ml job show with the expected flags."""
        from aml.cli import monitor_job
        captured: dict = {}

        class FakeResult:
            returncode = 0
            stdout = "Running\n"

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return FakeResult()

        monkeypatch.setattr(monitor_job.subprocess, "run", fake_run)
        status = monitor_job._poll_status("myjob", "myrg", "myws")
        assert status == "Running"
        cmd = captured["cmd"]
        assert "az" in cmd[0]
        assert "ml" in cmd
        assert "job" in cmd
        assert "--name" in cmd and cmd[cmd.index("--name") + 1] == "myjob"
        assert "--resource-group" in cmd and cmd[cmd.index("--resource-group") + 1] == "myrg"
        assert "--workspace-name" in cmd and cmd[cmd.index("--workspace-name") + 1] == "myws"

    def test_subprocess_failure_returns_unknown(self, monkeypatch):
        from aml.cli import monitor_job

        class FakeResult:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(monitor_job.subprocess, "run",
                            lambda *a, **kw: FakeResult())
        assert monitor_job._poll_status("j", "rg", "ws") == "Unknown"

    def test_az_not_installed_returns_sentinel(self, monkeypatch):
        from aml.cli import monitor_job

        def fake_run(*a, **kw):
            raise FileNotFoundError("az not found")

        monkeypatch.setattr(monitor_job.subprocess, "run", fake_run)
        assert monitor_job._poll_status("j", "rg", "ws") == "AzCliMissing"
