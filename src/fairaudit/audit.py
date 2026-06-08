"""Append-only chained NDJSON ledger (mirrors the portfolio substrate).

Each audit run writes its pre- and post-mitigation fairness summaries here, so the
measurement is itself auditable: editing any earlier entry breaks the hash chain.
"""
from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = Path("audit/local-demo.ndjson")
GENESIS = "GENESIS"


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _entry_hash(entry: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


def _prev_hash(ledger_path: Path) -> str:
    if not ledger_path.exists():
        return GENESIS
    last = None
    for line in ledger_path.read_text().splitlines():
        if line.strip():
            last = line
    return GENESIS if last is None else _entry_hash(json.loads(last))


def emit(action: str, job_id: str, fields: dict[str, Any] | None = None,
         *, ledger_path: Path | None = None) -> dict[str, Any]:
    ledger_path = ledger_path or DEFAULT_LEDGER
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
        "action": action,
        "actor": getpass.getuser(),
        "job_id": job_id,
        "fields": fields or {},
        "prev_hash": _prev_hash(ledger_path),
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(_canonical(entry).decode("utf-8") + "\n")
    return entry


def verify(ledger_path: Path | None = None) -> tuple[bool, int]:
    """Return (chain_ok, n_entries) by replaying the prev_hash chain."""
    ledger_path = ledger_path or DEFAULT_LEDGER
    if not ledger_path.exists():
        return True, 0
    prev, n = GENESIS, 0
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("prev_hash") != prev:
            return False, n
        prev = _entry_hash(entry)
        n += 1
    return True, n
