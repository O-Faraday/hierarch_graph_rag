"""Windows regression: text files must be UTF-8 whatever the locale (cp1252 cannot encode ∗ or 𝐷)."""

from __future__ import annotations

import json

from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size

from hgrag import pipeline


def test_hierarchy_written_in_utf8_even_with_cp1252_default(tmp_path, monkeypatch):
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))
    h = d.add_heading(text="Notation 𝐷 and ∗", level=1, prov=prov)
    d.add_text(label="text", text="Let 𝐷 denote the dataset and ∗ the convolution.", parent=h, prov=prov)
    d.save_as_json(tmp_path / "document.json")
    (tmp_path / "figures.json").write_text("[]", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"doc_id": "x", "source_pdf": "x.pdf"}), encoding="utf-8")

    # Simulate Windows: when no `encoding=` is given, pathlib/open fall back to cp1252.
    # (pathlib resolves the default through io.text_encoding.)
    import io

    real = io.text_encoding
    monkeypatch.setattr(io, "text_encoding", lambda enc, stacklevel=2: "cp1252" if enc is None else real(enc, stacklevel))

    out = pipeline.build_and_save_hierarchy(tmp_path)
    assert out["n_elements"] >= 1
    text = (tmp_path / "hierarchy.json").read_text(encoding="utf-8")
    assert "𝐷" in text and "∗" in text
