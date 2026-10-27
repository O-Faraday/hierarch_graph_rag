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


SECRET_NAMES = ("ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "OPENAI_API_KEY")


def load_config_file(path: Path) -> None:
    """Read the versioned settings file (TOML, any table nesting, keys = variable names) into os.environ.

    Never overrides a variable already set (environment or .env), so the order is: environment > .env > this
    file > code defaults. Booleans become "1"/"0", numbers their text. Secrets are refused: they only live in .env.
    """
    if not path.is_file():
        return
    import tomllib

    def flatten(table: dict) -> dict:
        out = {}
        for key, value in table.items():
            out.update(flatten(value) if isinstance(value, dict) else {key: value})
        return out

    values = flatten(tomllib.loads(path.read_text(encoding="utf-8-sig")))
    if secrets := sorted(set(values) & set(SECRET_NAMES)):
        raise ValueError(f"{path.name} is versioned: move {', '.join(secrets)} to .env")
    for key, value in values.items():
        if isinstance(value, bool):
            value = "1" if value else "0"
        os.environ.setdefault(key, str(value))


load_dotenv(PROJECT_ROOT / ".env")
CONFIG_FILE = os.environ.get("HGRAG_CONFIG", str(PROJECT_ROOT / "hgrag.toml"))  # "" = no settings file
if CONFIG_FILE:
    load_config_file(Path(CONFIG_FILE))

# Input PDFs (source to be defined later) and outputs.
PDF_DIR = Path(os.environ.get("HGRAG_PDF_DIR", PROJECT_ROOT / "data" / "pdf_sample"))
CONTROL_PDF_DIR = Path(os.environ.get("HGRAG_CONTROL_PDF_DIR", PROJECT_ROOT / "data" / "pdf_control"))
OUTPUT_DIR = Path(os.environ.get("HGRAG_OUTPUT_DIR", PROJECT_ROOT / "data" / "output"))
QDRANT_PATH = Path(os.environ.get("HGRAG_QDRANT_PATH", PROJECT_ROOT / "data" / "qdrant"))

# One single vector store (one collection) for all documents.
COLLECTION_NAME = os.environ.get("HGRAG_COLLECTION", "hgrag_chunks")

# English embedding model; the chunker tokenizer is aligned on it. bge-base-en-v1.5: 768 dimensions, 512-token window,
# far better on retrieval (MTEB) than all-MiniLM-L6-v2 (384 dimensions, 256 tokens). A different model means another
# vector size: index into a new collection / Qdrant folder (the ragbench does it by itself).
EMBED_MODEL_ID = os.environ.get("HGRAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# Chunks are cut to at most this many tokens (and never beyond the encoder window): the same chunks whatever the model.
CHUNK_MAX_TOKENS = int(os.environ.get("HGRAG_CHUNK_MAX_TOKENS", "256"))


# Qwen3-Embedding: one-sentence task description put before every query (in English, as the model card advises).
QWEN3_TASK = os.environ.get("HGRAG_QWEN3_TASK",
                            "Given a question about a scientific or technical document, retrieve the passages that answer it")


def embed_prefixes(model_id: str = EMBED_MODEL_ID) -> tuple[str, str]:
    """(query prefix, passage prefix) the model was trained with. Overridable: HGRAG_EMBED_QUERY_PREFIX / _DOC_PREFIX."""
    name = model_id.lower()
    if "bge-" in name and "-en" in name:  # bge-*-en-v1.5: instruction on the query only (model card)
        query, doc = "Represent this sentence for searching relevant passages: ", ""
    elif "e5-" in name or name.endswith("/e5"):
        query, doc = "query: ", "passage: "
    elif "qwen3-embedding" in name:  # "Instruct: {task}\nQuery:{query}" on the query only (model card)
        query, doc = f"Instruct: {QWEN3_TASK}\nQuery:", ""
    else:
        query, doc = "", ""
    return (os.environ.get("HGRAG_EMBED_QUERY_PREFIX", query), os.environ.get("HGRAG_EMBED_DOC_PREFIX", doc))


# Reranking: the vector search fetches RERANK_CANDIDATES chunks, a cross-encoder reads (question, chunk) pairs and the
# best k are kept. The cross-encoder sees both texts together, which a vector comparison cannot do.
RERANK = os.environ.get("HGRAG_RERANK", "1").strip().lower() in ("1", "true", "yes", "on")
RERANK_MODEL = os.environ.get("HGRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2")
RERANK_CANDIDATES = int(os.environ.get("HGRAG_RERANK_CANDIDATES", "30"))
RERANK_MAX_LENGTH = int(os.environ.get("HGRAG_RERANK_MAX_LENGTH", "512"))
# Hybrid search: share of the BM25 keyword score in the candidate ranking (min-max normalised scores, the rest is the
# dense similarity). 0 = dense only. Only for collections indexed with keyword vectors (keywords.has_keywords).
KEYWORD_WEIGHT = float(os.environ.get("HGRAG_KEYWORD_WEIGHT", "0.3"))

# Open RAG Bench questions whose reference answer was checked against the PDF and found wrong ({query id: reason}):
# the ragbench summary also reports the metrics without them ("excluding_disputed").
RAGBENCH_DISPUTED = Path(os.environ.get("HGRAG_RAGBENCH_DISPUTED", PROJECT_ROOT / "data" / "ragbench_disputed.json"))

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
# After the LaTeX, a plain-English reading of each formula (HGRAG_TABLE_MODEL, text only), added to the embedding text.
FORMULA_READING = _flag("HGRAG_FORMULA_READING", True)
FORMULA_ENRICHMENT = FORMULA_MODE == "docling"  # Docling's own enrichment (the model is run by Docling on CPU)
CODE_ENRICHMENT = _flag("HGRAG_CODE_ENRICHMENT", False)  # code blocks: language + indentation (CodeFormulaV2 on CPU: ~3 min per item)
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
# Second opinion on the low judge scores of a ragbench run (hgrag.refine): context judge on the whole PDF, then
# generation judge. Questions scored strictly below REFINE_THRESHOLD are judged again; the generation judge runs only
# when the context scores at least REFINE_CONTEXT_OK.
REFINE_MODEL = os.environ.get("HGRAG_REFINE_MODEL", "claude-haiku-5-5")
REFINE_THRESHOLD = int(os.environ.get("HGRAG_REFINE_THRESHOLD", "4"))
REFINE_CONTEXT_OK = int(os.environ.get("HGRAG_REFINE_CONTEXT_OK", "4"))

# Production assessments of the pipeline steps (hgrag.assessments): no reference answer involved. Deterministic checks
# always run; the LLM judges (figures and formulas, vision) run on a sample of the items.
ASSESS_MIN_COVERAGE = float(os.environ.get("HGRAG_ASSESS_MIN_COVERAGE", "0.9"))  # PDF text-layer words found by Docling
ASSESS_MIN_PAGE_COVERAGE = float(os.environ.get("HGRAG_ASSESS_MIN_PAGE_COVERAGE", "0.6"))  # same, page by page
ASSESS_MAX_HEADINGS_PER_PAGE = float(os.environ.get("HGRAG_ASSESS_MAX_HEADINGS_PER_PAGE", "4"))
ASSESS_MIN_DESCRIPTION_WORDS = int(os.environ.get("HGRAG_ASSESS_MIN_DESCRIPTION_WORDS", "15"))
ASSESS_JUDGE_MODEL = os.environ.get("HGRAG_ASSESS_JUDGE_MODEL", JUDGE_MODEL)
ASSESS_SAMPLE = float(os.environ.get("HGRAG_ASSESS_SAMPLE", "0.2"))  # share of the figures / formulas judged
ASSESS_MAX_JUDGED = int(os.environ.get("HGRAG_ASSESS_MAX_JUDGED", "10"))  # per document and per kind
ASSESS_JUDGE_OK = int(os.environ.get("HGRAG_ASSESS_JUDGE_OK", "4"))  # judge score (1-5) counted as passed
ASSESS_SMALL_CHUNK_WORDS = int(os.environ.get("HGRAG_ASSESS_SMALL_CHUNK_WORDS", "15"))  # a "small" chunk
ASSESS_MAX_SMALL_CHUNKS = float(os.environ.get("HGRAG_ASSESS_MAX_SMALL_CHUNKS", "0.4"))  # share of small chunks allowed
ASSESS_MAX_DUPLICATES = int(os.environ.get("HGRAG_ASSESS_MAX_DUPLICATES", "3"))  # repeated chunks allowed per document
ASSESS_MIN_TOP_SCORE = float(os.environ.get("HGRAG_ASSESS_MIN_TOP_SCORE", "0.1"))  # best reranker score of a question
ASSESS_MIN_AGREEMENT = float(os.environ.get("HGRAG_ASSESS_MIN_AGREEMENT", "0.1"))  # candidates found by both searches
# Context judge: a sufficiency of 2 or less fails. At 4, as the other judges, it flagged 27 % of the ragbench questions
# with only 23 % of bad answers among them; at 3, 7 % with 35 % (base rate 15 %).
ASSESS_MIN_SUFFICIENCY = int(os.environ.get("HGRAG_ASSESS_MIN_SUFFICIENCY", "3"))
ASSESS_MAX_UNSOURCED_NUMBERS = int(os.environ.get("HGRAG_ASSESS_MAX_UNSOURCED_NUMBERS", "0"))  # per answer
# MLflow traces (hgrag.tracing): inputs and outputs of each step, when MLflow tracking is on (a run_key is given).
TRACING = _flag("HGRAG_TRACING", True)
# LLM judges of the production checks inside the pipeline and the app (cost: API calls). Off: deterministic checks only.
ASSESS_JUDGES = _flag("HGRAG_ASSESS_JUDGES", False)
