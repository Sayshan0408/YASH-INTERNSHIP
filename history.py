"""
core/history.py
Persistent run history stored in data/history.json.
Keeps the 50 most recent runs automatically.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_HISTORY_FILE = os.path.join(_DATA_DIR, "history.json")
_MAX_RUNS = 50


def _ensure_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _load() -> List[dict]:
    _ensure_dir()
    if not os.path.exists(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _persist(runs: List[dict]) -> None:
    _ensure_dir()
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)


def save_run(requirement: str, outputs: Dict[str, str], status: str = "success") -> dict:
    """Persist a completed pipeline run. Returns the saved run record."""
    runs = _load()
    now = datetime.now(timezone.utc)
    run: dict = {
        "id": now.strftime("%Y%m%d%H%M%S%f"),
        "timestamp": now.isoformat(),
        "requirement": requirement,
        "outputs": outputs,
        "status": status,
        "stage_count": len(outputs),
    }
    runs.insert(0, run)
    _persist(runs[:_MAX_RUNS])
    return run


def list_runs() -> List[dict]:
    """Return all saved runs, newest first."""
    return _load()


def search_runs(keyword: str) -> List[dict]:
    """Return runs whose requirement contains keyword (case-insensitive)."""
    if not keyword or not keyword.strip():
        return _load()
    kw = keyword.strip().lower()
    return [r for r in _load() if kw in r.get("requirement", "").lower()]


def get_run(run_id: str) -> Optional[dict]:
    """Return a single run by its ID, or None."""
    return next((r for r in _load() if r.get("id") == run_id), None)


def delete_run(run_id: str) -> bool:
    """Delete a run by ID. Returns True if deleted."""
    runs = _load()
    filtered = [r for r in runs if r.get("id") != run_id]
    if len(filtered) == len(runs):
        return False
    _persist(filtered)
    return True


def clear_all_runs() -> None:
    """Delete all history."""
    _persist([])
