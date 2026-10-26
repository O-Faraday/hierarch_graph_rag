"""Log the two applications as MLflow pyfunc models and register them (``hgrag register``).

Each model is logged "from code" (``model_code/<kind>.py``) with the requirements exported from ``uv.lock`` (the
``hgrag`` package itself comes from the project environment, ``uv sync``). Its params are the configuration of the
version (``versions``) and its tag the hash of that configuration, so the traces made with the same configuration are
linked to it. Registering again with an unchanged configuration reuses the model already logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config, tracking, versions

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_CODE = {kind: PACKAGE_DIR / "model_code" / f"{kind}.py" for kind in versions.KINDS}


def default_model_config(kind: str) -> dict[str, Any]:
    if kind == "rag":
        return {"output_dir": str(config.OUTPUT_DIR), "qdrant_path": str(config.QDRANT_PATH),
                "collection": config.COLLECTION_NAME, "embed_model": config.EMBED_MODEL_ID, "k": 10,
                "judges": config.ASSESS_JUDGES}
    return {"describe_pictures": config.DESCRIBE_PICTURES, "steps": "parse,hierarchy,index"}


def register(kind: str, *, model_config: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Log and register ``kind`` (``rag`` or ``indexing``); returns model id, registered name and version."""
    import mlflow

    name = versions.MODEL_NAMES[kind]
    tracking._setup(versions.EXPERIMENTS[kind])
    mlflow.set_experiment(versions.EXPERIMENTS[kind])  # log_model uses the active experiment (no live span here)
    existing = versions.model_id(kind)
    if existing and not force:
        found = mlflow.MlflowClient().search_model_versions(f"name = '{name}'")
        # the SQLite registry may leave model_id empty; the source is models:/<model id>
        version = next((int(v.version) for v in found if existing in (v.model_id or "", v.source or "")
                        or (v.source or "").endswith(existing)), None)
        return {"kind": kind, "model_id": existing, "name": name, "version": version, "reused": True}
    # No code_paths: the hgrag package comes from the project environment (uv.lock and pyproject.toml are copied with
    # the model). Copying the package as code_paths fails on Windows: MLflow cannot remove its temporary copy (WinError 5).
    info = mlflow.pyfunc.log_model(
        name=name, python_model=str(MODEL_CODE[kind]), model_config=model_config or default_model_config(kind),
        uv_project_path=str(config.PROJECT_ROOT), registered_model_name=name, params=versions.config_snapshot(kind),
        tags={versions.HASH_TAG: versions.config_hash(kind)})
    versions._IDS.clear()
    return {"kind": kind, "model_id": info.model_id, "name": name,
            "version": getattr(info, "registered_model_version", None), "reused": False}
