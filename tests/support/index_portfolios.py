# ibexQuant\tests\support\index_portfolios.py

"""Small B3 index portfolio workbooks, in the old and new layouts (ADR 0015)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import openpyxl

# (ticker, name, share class, theoretical quantity, weight %)
Member = tuple[str, str, str, float, float]

OLD_HEADER = ("COD.", "ACAO", "TIPO", "QTDE. TEORICA", "PART. %")
NEW_HEADER = ("CÓDIGO", "AÇÃO", "TIPO", "QTDE. TEÓRICA", "PART. %")


def write_workbook(
    path: Path,
    title: str,
    sheets: dict[str, list[Member]],
    *,
    base: dt.date | None = dt.date(2024, 8, 28),
    old_layout: bool = False,
    no_base_in: frozenset[str] = frozenset(),
) -> Path:
    """Write one workbook; each sheet gets ``title`` in A1 and a footer with ``base``."""
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, members in sheets.items():
        sheet = book.create_sheet(name)
        sheet.append([title])
        sheet.append(list(OLD_HEADER if old_layout else NEW_HEADER))
        for member in members:
            sheet.append(list(member))
        total = sum(m[3] for m in members if isinstance(m[3], int | float))
        if old_layout:
            sheet.append(["QUANTIDADE TEORICA TOTAL", None, None, total, 100])
            sheet.append(["VALOR DO REDUTOR", None, None, 1.5, None])
        else:
            sheet.append([None, "QUANTIDADE TEORICA TOTAL", None, total, 100])
            sheet.append([None, "VALOR DO REDUTOR", None, 1.5, None])
        sheet.append([])
        if base is not None and name not in no_base_in:
            if old_layout:
                sheet.append([f"PREGÃO BASE: {base:%d/%m/%Y}"])
            else:
                sheet.append(["Pregão Base:", dt.datetime.combine(base, dt.time())])
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path
