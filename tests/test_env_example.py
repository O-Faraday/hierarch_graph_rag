"""env.example (secrets of .env) and hgrag.toml (versioned settings): they parse, hold no secret, name only variables
the code reads."""

import os
import re
import tomllib
from pathlib import Path

import pytest

from hgrag import config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "env.example"
SETTINGS = ROOT / "hgrag.toml"
THIRD_PARTY = {"HF_HUB_OFFLINE", "MLFLOW_DISABLE_TELEMETRY", "MLFLOW_TRACKING_URI"}  # read by the libraries


def _code() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for d in ("src", "app") for p in (ROOT / d).rglob("*.py"))


def test_env_example_holds_only_empty_secrets(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    config.load_dotenv(EXAMPLE)
    assert os.environ["ANTHROPIC_API_KEY"] == "" and os.environ["HF_TOKEN"] == ""
    active = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", EXAMPLE.read_text(encoding="utf-8"), re.M))
    assert active <= set(config.SECRET_NAMES)


def test_settings_file_loads_and_holds_no_secret(monkeypatch):
    for key in ("HGRAG_EMBED_MODEL", "HGRAG_ASSESS_JUDGES", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    config.load_config_file(SETTINGS)
    assert os.environ["HGRAG_EMBED_MODEL"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert os.environ["HGRAG_ASSESS_JUDGES"] == "0" and os.environ["HF_HUB_OFFLINE"] == "1"
    assert not re.search(r"(KEY|TOKEN|SECRET|PASSWORD)\s*=", SETTINGS.read_text(encoding="utf-8"))


def test_config_file_never_overrides_and_refuses_secrets(tmp_path, monkeypatch):
    f = tmp_path / "s.toml"
    f.write_text('[a]\nHGRAG_X = true\n[b.c]\nHGRAG_Y = 0.5\nEXISTING = "file"\n', encoding="utf-8")
    for k in ("HGRAG_X", "HGRAG_Y"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EXISTING", "shell")
    config.load_config_file(f)
    assert (os.environ["HGRAG_X"], os.environ["HGRAG_Y"], os.environ["EXISTING"]) == ("1", "0.5", "shell")
    f.write_text('[x]\nHF_TOKEN = "hf_abc"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        config.load_config_file(f)


def test_every_variable_of_env_example_and_settings_is_read_by_the_code():
    names = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)\s*=", EXAMPLE.read_text(encoding="utf-8"), re.M))
    names |= set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)\s*=", SETTINGS.read_text(encoding="utf-8"), re.M))
    # Every commented default stays valid TOML once uncommented.
    tomllib.loads(re.sub(r"^# ([A-Z][A-Z0-9_]+ = )", r"\1", SETTINGS.read_text(encoding="utf-8"), flags=re.M))
    code = _code()
    assert names and not {n for n in names - THIRD_PARTY if f'"{n}"' not in code}
