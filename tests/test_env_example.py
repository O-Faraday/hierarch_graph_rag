"""env.example documents the .env file: it parses, holds no secret and names only variables the code reads."""

import os
import re
from pathlib import Path

from hgrag import config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "env.example"


def test_env_example_parses_and_holds_no_secret(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "HF_TOKEN", "HGRAG_EMBED_MODEL", "HGRAG_ASSESS_JUDGES"):
        monkeypatch.delenv(key, raising=False)
    config.load_dotenv(EXAMPLE)
    assert os.environ["ANTHROPIC_API_KEY"] == "" and os.environ["HF_TOKEN"] == ""
    assert os.environ["HGRAG_EMBED_MODEL"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert os.environ["HGRAG_ASSESS_JUDGES"] == "0"


def test_every_variable_of_env_example_is_read_by_the_code():
    names = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", EXAMPLE.read_text(encoding="utf-8"), re.M))
    code = "\n".join(p.read_text(encoding="utf-8") for d in ("src", "app") for p in (ROOT / d).rglob("*.py"))
    third_party = {"HF_HUB_OFFLINE", "MLFLOW_DISABLE_TELEMETRY", "MLFLOW_TRACKING_URI"}  # read by the libraries
    assert names and not {n for n in names - third_party if f'"{n}"' not in code}
