"""Lightweight Panel stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Panel:
    renderable: str
    title: str | None = None
    expand: bool = False
