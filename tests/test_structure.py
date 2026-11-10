"""structure.json: the readable tree of a document (title, sections, typed content, pages), next to hierarchy.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hgrag import cli, config, evaluation, pipeline, structure


def _el(ref, kind, text=None, page=1, **extra):
    return {"ref": ref, "kind": kind, "label": kind, "page_no": page, "text": text, **({"extra": extra} if extra else {})}


def _node(ref, title, level, page, elements=(), children=()):
    return {"ref": ref, "title": title, "level": level, "page_no": page, "path": title, "is_leaf": not children,
            "elements": list(elements), "children": list(children)}


TREE = _node("#/root", "report", 0, None, children=[
    _node("#/texts/0", "Signalling Standard SR-12", 1, 1, [_el("#/texts/1", "paragraph", "Regional Rail Authority")]),
    _node("#/texts/2", "2 Equipment", 1, 1, children=[
        _node("#/texts/3", "2.1 Barriers", 2, 1, [
            _el("#/texts/4", "paragraph", "The barriers are lowered."),
            _el("#/texts/5", "list_item", "three lamps", 1), _el("#/texts/6", "list_item", "one bell", 2),
            _el("#/tables/0", "table", "| a | b |", 2, caption="Table 1", num_rows=61, num_cols=4, pages=[2, 3],
                description="Torques."),
            _el("#/pictures/0", "figure", "A bar chart.", 3, caption="Figure 1", png_path="figures\\figure-001.png",
                description="A bar chart."),
            _el("#/texts/7", "formula", "M = sum", 3)])]),
    _node("#/texts/8", "Appendix A. Test equipment", 1, 4, [_el("#/texts/9", "paragraph", "A stopwatch.", 4)]),
])


def test_tree_with_title_sections_typed_content_and_pages():
    s = structure.to_structure(TREE, doc_id="d1", source_pdf="data\\pdf_eval\\sr12.pdf", n_pages=4,
                               formulas={"#/texts/7": {"latex": r"M = \sum", "reading": {"text": "The maturity."}}})
    assert (s["doc_id"], s["title"], s["source_pdf"], s["pages"]) == ("d1", "Signalling Standard SR-12", "sr12.pdf", 4)
    assert s["content"] == [{"type": "paragraph", "text": "Regional Rail Authority", "page": 1, "ref": "#/texts/1"}]
    equipment, appendix = s["sections"]
    assert (equipment["id"], equipment["number"], equipment["pages"]) == ("s1", "2", {"start": 1, "end": 3})
    assert (appendix["id"], appendix["number"], appendix["pages"]) == ("s2", "A", {"start": 4, "end": 4})
    barriers = equipment["sections"][0]
    assert (barriers["id"], barriers["number"], barriers["level"]) == ("s1.1", "2.1", 2)
    paragraph, lst, table, figure, formula = barriers["content"]
    assert lst == {"type": "list", "items": ["three lamps", "one bell"], "pages": [1, 2], "refs": ["#/texts/5", "#/texts/6"]}
    assert table == {"type": "table", "caption": "Table 1", "markdown": "| a | b |", "description": "Torques.", "rows": 61,
                     "cols": 4, "page": 2, "pages": [2, 3], "ref": "#/tables/0"}
    assert figure["image"] == "figures/figure-001.png" and figure["description"] == "A bar chart."
    assert (formula["latex"], formula["reading"], formula["page"]) == (r"M = \sum", "The maturity.", 3)


def test_a_numbered_or_standard_first_heading_is_not_the_title():
    for first in ("1 Introduction", "Abstract"):
        tree = _node("#/root", "paper", 0, None, children=[_node("#/texts/0", first, 1, 1)])
        s = structure.to_structure(tree, doc_id="d")
        assert s["title"] == "paper" and s["sections"][0]["title"] == first


def test_heading_numbers():
    assert [structure.heading_number(t) for t in ("2.1. Rainfall records", "2.1.1 Closing time", "IV. Results",
                                                   "B. Bacteria", "Appendix B: Glossary", "Purpose",
                                                   "A paper about trains")] == ["2.1", "2.1.1", "IV", "B", "B", None, None]


def test_parsed_documents_get_structure_json_and_the_cli_gathers_the_corpus(tmp_path, monkeypatch, capsys):
    out = tmp_path / "output"
    for name in ("01_aurora_bank_annual_report.pdf", "03_network_operations_handbook.pdf"):
        pdf = config.CONTROL_PDF_DIR / name
        dst = out / evaluation.compute_doc_id(pdf)
        shutil.copytree(config.CONTROL_PDF_DIR / "parsed" / dst.name, dst)
        pipeline.build_and_save_hierarchy(dst)  # writes hierarchy.json and structure.json
        s = json.loads((dst / structure.FILE).read_text(encoding="utf-8"))
        assert s["title"] and s["sections"] and all(sec["pages"] for sec in s["sections"])
    monkeypatch.setattr("sys.argv", ["hgrag", "structure", "--output-dir", str(out)])
    cli.main()
    docs = json.loads((out / structure.FILE).read_text(encoding="utf-8"))
    assert [d["source_pdf"] for d in docs] == ["01_aurora_bank_annual_report.pdf", "03_network_operations_handbook.pdf"]
    assert "2 document(s)" in capsys.readouterr().out
    titles = [sec["title"] for sec in docs[0]["sections"]]
    assert "2 Financial results" in titles
    revenue = next(sec for sec in docs[0]["sections"] if sec["title"] == "2 Financial results")["sections"][0]
    assert revenue["title"] == "2.1 Revenue" and {b["type"] for b in revenue["content"]} >= {"paragraph", "table"}
