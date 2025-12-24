"""Lightweight Table stub."""

from __future__ import annotations

from typing import List


class Table:
    def __init__(
        self,
        title: str | None = None,
        expand: bool | None = None,
        show_lines: bool = False,
        show_header: bool = True,
    ) -> None:  # noqa: D401 - mimic rich API
        self.title = title
        self.expand = expand
        self.show_lines = show_lines
        self.show_header = show_header
        self.columns: List[str] = []
        self.rows: List[list[str]] = []

    def add_column(self, header: str) -> None:
        self.columns.append(header)

    def add_row(self, *values: str) -> None:
        self.rows.append([str(v) for v in values])
