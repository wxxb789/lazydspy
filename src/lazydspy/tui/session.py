"""Session export/import helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

from lazydspy.tui.viewmodel import ViewModel


def export_session_json(viewmodel: ViewModel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_session_payload(viewmodel)
    path = output_dir / "session.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_session_jsonl(viewmodel: ViewModel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "session.jsonl"
    payload = _build_session_payload(viewmodel)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_session_json(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj).__name__}")
    return cast(dict[str, object], obj)


def load_session_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict, got {type(obj).__name__}")
            records.append(cast(dict[str, object], obj))
    return records


def _build_session_payload(viewmodel: ViewModel) -> dict[str, object]:
    payload = viewmodel.to_session_dict()
    payload["exported_at"] = datetime.utcnow().isoformat()
    return payload


__all__ = [
    "export_session_json",
    "export_session_jsonl",
    "load_session_json",
    "load_session_jsonl",
]
