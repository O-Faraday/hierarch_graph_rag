"""Document identifiers (lightweight: no Docling import)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def compute_doc_id(pdf_path: str | Path) -> str:
    """Content hash of the PDF: renaming a file keeps its id, editing it changes it."""
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()[:16]


def pdf_name(path: str) -> str:
    """File name of a path written on any OS (outputs produced on Windows are read on Linux too)."""
    return re.split(r"[\\/]", str(path).rstrip("\\/"))[-1]


def pdf_stem(path: str) -> str:
    name = pdf_name(path)
    return name.rsplit(".", 1)[0] if "." in name else name
