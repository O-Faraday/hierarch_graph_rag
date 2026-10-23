"""Production assessments of the pipeline steps: no reference answer is involved.

Each check returns ``Assessment`` records (name, value, pass/fail, rationale, the item it is about). Deterministic
checks (``source="code"``) always run; LLM judges (``source="llm_judge"``) run on a sample of the items. A record
maps one to one onto an MLflow ``Feedback`` (``to_feedback``), to attach to the trace of the step that produced the
item. The records of a document are also written to ``assessments.json`` next to its Docling outputs.

    uv run hgrag assess --output-dir data/output            # deterministic checks
    uv run hgrag assess --output-dir data/output --judge    # + LLM judges on a sample
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import config

STEPS = ("parse", "describe", "hierarchy", "chunks")


@dataclass
class Assessment:
    name: str  # "<step>.<check>", e.g. parse.text_coverage
    value: Any  # number, bool or str (what MLflow accepts as a feedback value); None on error
    source: str = "code"  # "code" (deterministic) or "llm_judge"
    passed: bool | None = None  # None: informative only, or not computable
    rationale: str = ""
    target: str | None = None  # the item assessed (Docling self_ref, "page 3"); None = the whole document
    source_id: str = "hgrag"  # check function or judge model
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_feedback(self, trace_id: str | None = None, span_id: str | None = None) -> Any:
        """The record as an MLflow ``Feedback`` (metadata values are strings, as MLflow stores them)."""
        from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

        kind = AssessmentSourceType.LLM_JUDGE if self.source == "llm_judge" else AssessmentSourceType.CODE
        meta = {**self.metadata, "passed": self.passed, "target": self.target}
        meta = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in meta.items() if v is not None}
        return Feedback(name=self.name, value=self.value, error=self.error, rationale=self.rationale or None,
                        source=AssessmentSource(source_type=kind, source_id=self.source_id), metadata=meta,
                        trace_id=trace_id, span_id=span_id)


def sampled(key: str, rate: float) -> bool:
    """Deterministic sampling: the same item is always in (or out of) the sample at a given rate."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def assess_document(doc_dir: str | Path, *, steps: tuple[str, ...] = STEPS, judge: bool = False,
                    sample: float | None = None, max_judged: int | None = None, judge_model: str | None = None,
                    api_key: str | None = None, workers: int = 4, write: bool = True, client: Any = None,
                    collection: str | None = None) -> dict[str, list[Assessment]]:
    """Run the checks of ``steps`` on one parsed document; with ``write`` they replace the same steps in
    ``assessments.json`` (the other steps are kept). ``client``/``collection``: the Qdrant store the document was
    indexed into (step "chunks")."""
    from docling_core.types.doc import DoclingDocument

    from . import chunks as chunk_checks
    from . import description, hierarchy, parsing

    doc_dir = Path(doc_dir)
    doc = DoclingDocument.load_from_json(doc_dir / "document.json")
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    figures = _load_list(doc_dir / "figures.json")
    out: dict[str, list[Assessment]] = {}
    if "parse" in steps:
        out["parse"] = parsing.assess(doc, figures, doc_dir, pdf_path=manifest.get("source_pdf"))
    if "describe" in steps:
        out["describe"] = description.assess(
            doc, figures, _load_list(doc_dir / "tables.json"), _load_list(doc_dir / "formulas.json"), doc_dir,
            judge=judge, sample=config.ASSESS_SAMPLE if sample is None else sample,
            max_judged=config.ASSESS_MAX_JUDGED if max_judged is None else max_judged,
            judge_model=judge_model, api_key=api_key, workers=workers)
    if "hierarchy" in steps:
        path = doc_dir / "hierarchy.json"
        out["hierarchy"] = (hierarchy.assess(json.loads(path.read_text(encoding="utf-8")), judge=judge,
                                             judge_model=judge_model, api_key=api_key) if path.exists()
                            else [Assessment("hierarchy.numbering", None, error="hierarchy.json not found")])
    if "chunks" in steps:
        path = doc_dir / "chunks.jsonl"
        if path.exists():
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
            out["chunks"] = chunk_checks.assess(rows, manifest["doc_id"], client=client, collection=collection)
        else:
            out["chunks"] = [Assessment("chunks.size_distribution", None, error="chunks.jsonl not found: not indexed")]
    if write:
        path = doc_dir / "assessments.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        saved.update({step: [a.to_dict() for a in recs] for step, recs in out.items()})
        path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_assessments(doc_dir: str | Path) -> dict[str, list[Assessment]]:
    path = Path(doc_dir) / "assessments.json"
    if not path.exists():
        return {}
    return {step: [Assessment(**r) for r in recs] for step, recs in json.loads(path.read_text(encoding="utf-8")).items()}


def summarize(records: list[Assessment]) -> list[dict[str, Any]]:
    """One line per check over many records: count, passed, failed, errors, mean of the numeric values."""
    out = []
    for name in sorted({r.name for r in records}):
        mine = [r for r in records if r.name == name]
        nums = [r.value for r in mine if isinstance(r.value, (int, float)) and not isinstance(r.value, bool)]
        out.append({"name": name, "source": mine[0].source, "n": len(mine),
                    "passed": sum(r.passed is True for r in mine), "failed": sum(r.passed is False for r in mine),
                    "errors": sum(r.error is not None for r in mine),
                    "mean": round(statistics.fmean(nums), 3) if nums else None})
    return out


def _load_list(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
