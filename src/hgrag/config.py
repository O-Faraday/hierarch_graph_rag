"""Central configuration (paths and model identifiers)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HGRAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def load_dotenv(path: Path) -> None:
    """Read ``KEY=value`` lines of a .env file into os.environ (never overriding a variable already set).

    Done here, before anything else, so HF_HUB_OFFLINE and ANTHROPIC_API_KEY are seen by every entry point
    (CLI, app, runner subprocess) and before huggingface_hub is imported.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            os.environ.setdefault(key, value)


load_dotenv(PROJECT_ROOT / ".env")

# Input PDFs (source to be defined later) and outputs.
PDF_DIR = Path(os.environ.get("HGRAG_PDF_DIR", PROJECT_ROOT / "data" / "pdf_sample"))
CONTROL_PDF_DIR = Path(os.environ.get("HGRAG_CONTROL_PDF_DIR", PROJECT_ROOT / "data" / "pdf_control"))
OUTPUT_DIR = Path(os.environ.get("HGRAG_OUTPUT_DIR", PROJECT_ROOT / "data" / "output"))
QDRANT_PATH = Path(os.environ.get("HGRAG_QDRANT_PATH", PROJECT_ROOT / "data" / "qdrant"))

# One single vector store (one collection) for all documents.
COLLECTION_NAME = os.environ.get("HGRAG_COLLECTION", "hgrag_chunks")

# English embedding model; the chunker tokenizer is aligned on it.
EMBED_MODEL_ID = os.environ.get("HGRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paragraph chunks with fewer words than this are dropped (chart labels, stray words...).
MIN_CHUNK_WORDS = int(os.environ.get("HGRAG_MIN_CHUNK_WORDS", "3"))
# Prose chunks (text, paragraph, list item, footnote) with fewer words than this are dropped as well: a date line or an
# author name carries no content, and with the heading prepended it would win every search on the document title.
MIN_PROSE_WORDS = int(os.environ.get("HGRAG_MIN_PROSE_WORDS", "5"))
# Number of closest headings prepended to the text that is embedded (0 = none, 99 = the whole chain). The document
# title (a Docling "title" item) is never prepended: it would make every chunk of the front matter (affiliations, dates)
# match any query on the topic of the paper.
HEADING_CONTEXT = int(os.environ.get("HGRAG_HEADING_CONTEXT", "99"))
# Sections that are not indexed (bibliography, acknowledgements): their entries win searches on author names and topics.
EXCLUDED_SECTIONS = os.environ.get(
    "HGRAG_EXCLUDED_SECTIONS", r"^\W*(?:[0-9ivx]+[.)]?\s+)?(references|bibliography|acknowledg(?:e)?ments?)\W*$")

# Docling parsing options (docling 2.138 defaults: table structure on, ACCURATE, cell matching on; enrichments off).
def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


TABLE_MODE = os.environ.get("HGRAG_TABLE_MODE", "accurate").strip().lower()  # accurate | fast
TABLE_CELL_MATCHING = _flag("HGRAG_TABLE_CELL_MATCHING", True)
# Formulas -> LaTeX: "vlm" = vision model through the HF API after the parsing (default, seconds per formula),
# "docling" = Docling's CodeFormulaV2 on CPU (about a minute per formula), "off".
FORMULA_MODE = os.environ.get("HGRAG_FORMULA_MODE", "vlm").strip().lower()
FORMULA_VLM = os.environ.get("HGRAG_FORMULA_VLM", "")  # empty = the figure model
FORMULA_ENRICHMENT = FORMULA_MODE == "docling"  # Docling's own enrichment (the model is run by Docling on CPU)
CODE_ENRICHMENT = _flag("HGRAG_CODE_ENRICHMENT", True)  # code blocks: language + indentation
PICTURE_CLASSIFICATION = _flag("HGRAG_PICTURE_CLASSIFICATION", False)  # figure type (chart, diagram...) in the metadata
TABLE_IMAGES = _flag("HGRAG_TABLE_IMAGES", False)  # PNG of each table (to describe a table from its picture)

# Figure export / description.
IMAGES_SCALE = float(os.environ.get("HGRAG_IMAGES_SCALE", "2.0"))
PICTURE_PROMPT = os.environ.get("HGRAG_PICTURE_PROMPT", "")  # empty = the prompt validated by `hgrag vlm-bench`

# Figures are described AFTER parsing by a vision model called through an API (nothing is installed locally).
# Winner of `hgrag vlm-bench` on the control figures: score 5/5, every fact found, nothing invented.
FIGURE_VLM = os.environ.get("HGRAG_FIGURE_VLM", "hf:Qwen/Qwen3-VL-30B-A3B-Instruct")
# Tables: described from their Markdown (text only), the description is added to the embedding text of the chunks.
TABLE_MODEL = os.environ.get("HGRAG_TABLE_MODEL", FIGURE_VLM)
# On by default as soon as an HF token is available; HGRAG_DESCRIBE_PICTURES=0/1 forces it.
DESCRIBE_PICTURES = os.environ.get(
    "HGRAG_DESCRIBE_PICTURES", "1" if os.environ.get("HF_TOKEN") else "0").lower() in ("1", "true", "yes")

# Docling labels that become chunks. Default: every textual element (paragraphs, list items, footnotes,
# formulas, code, tables, captions). Figures themselves are not chunked (their description is, see indexing.figure_chunks).
CHUNK_LABELS = tuple(
    x.strip() for x in os.environ.get(
        "HGRAG_CHUNK_LABELS", "text,paragraph,list_item,footnote,formula,code,table,caption").split(",") if x.strip()
)

# Evaluation: frozen Docling outputs of the control PDFs, and the LLM judge.
EVAL_DIR = Path(os.environ.get("HGRAG_EVAL_DIR", PROJECT_ROOT / "data" / "eval"))
JUDGE_MODEL = os.environ.get("HGRAG_JUDGE_MODEL", "claude-haiku-5-5")
