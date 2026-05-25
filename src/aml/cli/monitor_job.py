"""Monitor an AML job: poll status, emit one structured event per transition.

Runs via the operator's Monitor tool to get push notifications on job state
changes (Running → Completed / Failed / Canceled). Each stdout line is one
JSON event the Monitor tool consumes; one notification per transition.

Uses the `az` CLI rather than the azure.ai.ml SDK so credentials auto-refresh
during long polls — the SDK has a known 12-hour token expiry issue that
bit us during the Stage 5 hotfix.

Run:
    # Foreground, emits events until job reaches terminal state
    python -m aml.cli.monitor_job --job-name jolly_tangelo_d36rg40785

    # With explicit poll interval (default 30s; cap = 600s)
    python -m aml.cli.monitor_job --job-name X --poll-seconds 60

    # Via Monitor tool (operator workflow)
    # The Monitor tool sees each stdout JSON line as an event and emits
    # one notification per line.

Terminal states: Completed, Failed, Canceled, NotResponding.

Exit code:
    0 if job reaches Completed
    1 if Failed / Canceled / monitor errored
    The CLI never hangs — even on NotResponding it emits an event and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

TERMINAL_STATES = {"Completed", "Failed", "Canceled", "NotResponding"}
PASS_STATES = {"Completed"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _emit(event: dict) -> None:
    """Print one JSON line; flush so the Monitor tool sees it immediately."""
    line = json.dumps(event, separators=(",", ":"))
    print(line, flush=True)


def _poll_status(job_name: str, rg: str, ws: str) -> str:
    """Run `az ml job show --query status` and return the status string.

    Returns "Unknown" on any subprocess error so the monitor doesn't crash
    on a transient CLI hiccup.
    """
    cmd = [
        "az", "ml", "job", "show",
        "--name", job_name,
        "--resource-group", rg,
        "--workspace-name", ws,
        "--query", "status",
        "-o", "tsv",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Unknown"
    except FileNotFoundError:
        # `az` not installed — surface a clean error
        return "AzCliMissing"
    if p.returncode != 0:
        return "Unknown"
    return p.stdout.strip() or "Unknown"


def monitor(
    job_name: str,
    rg: str,
    ws: str,
    poll_seconds: int,
    max_wait_seconds: int | None = None,
) -> int:
    """Poll job status until terminal; emit one event per transition + summary.

    Returns 0 if job ended in PASS_STATES, 1 otherwise.
    """
    last_status = None
    started = time.monotonic()
    _emit({
        "ts": _utc_now_iso(), "event": "monitor_start",
        "job": job_name, "poll_seconds": poll_seconds,
    })
    while True:
        status = _poll_status(job_name, rg, ws)
        if status == "AzCliMissing":
            _emit({"ts": _utc_now_iso(), "event": "error",
                   "message": "az CLI not found on PATH"})
            return 1
        if status != last_status:
            _emit({
                "ts": _utc_now_iso(), "event": "status_transition",
                "job": job_name, "from": last_status, "to": status,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
            last_status = status
        if status in TERMINAL_STATES:
            elapsed = round(time.monotonic() - started, 2)
            _emit({
                "ts": _utc_now_iso(), "event": "monitor_end",
                "job": job_name, "final_status": status,
                "elapsed_seconds": elapsed,
            })
            return 0 if status in PASS_STATES else 1
        if max_wait_seconds is not None and (time.monotonic() - started) > max_wait_seconds:
            _emit({
                "ts": _utc_now_iso(), "event": "monitor_timeout",
                "job": job_name, "last_status": status,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
            return 1
        time.sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job-name", required=True, help="AML job name (e.g. jolly_tangelo_d36rg40785)")
    ap.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP", ""),
                    help="Defaults to $AZURE_RESOURCE_GROUP")
    ap.add_argument("--workspace-name", default=os.environ.get("AZUREML_WORKSPACE_NAME", "bwm-ml01-train-wus3"),
                    help="Defaults to $AZUREML_WORKSPACE_NAME or bwm-ml01-train-wus3")
    ap.add_argument("--poll-seconds", type=int, default=30,
                    help="Polling interval; clamped to [10, 600] (default: 30)")
    ap.add_argument("--max-wait-seconds", type=int, default=None,
                    help="Optional overall timeout; default unlimited")
    args = ap.parse_args()

    if not args.resource_group:
        print("error: --resource-group required (or set AZURE_RESOURCE_GROUP)",
              file=sys.stderr)
        return 2

    poll = max(10, min(600, args.poll_seconds))
    return monitor(
        args.job_name, args.resource_group, args.workspace_name,
        poll, max_wait_seconds=args.max_wait_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
