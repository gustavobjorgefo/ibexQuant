# ibexQuant\src\ibexQuant\data\ingestion\cvm\__init__.py

"""CVM open data: the ticker-to-CNPJ table of the registration forms (ADR 0012)."""

from __future__ import annotations

from ibexQuant.data.ingestion.cvm.fca import FcaFormatError, fca_files, read_fca_securities

__all__ = ["FcaFormatError", "fca_files", "read_fca_securities"]
