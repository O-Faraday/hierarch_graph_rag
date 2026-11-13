# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluate the index: the questions of an evaluation dataset, scored like `hgrag eval`
# MAGIC Thin notebook: the logic is in `hgrag.dbx.evaluate`. Run `hgrag_ingest` first.
# MAGIC The dataset is a JSON Lines file on the volume (`eval/eval_dataset.jsonl` by default, the format of
# MAGIC `data/pdf_eval/eval_dataset.jsonl`). Each question is searched in AI Search, answered and judged against its reference
# MAGIC answer with the Claude API (`ANTHROPIC_API_KEY` secret of the scope). The run is written to `eval/runs/<run>/` on the
# MAGIC volume: `results.jsonl`, `summary.json` and `report.md`, the same files as a local run.

# COMMAND ----------

dbutils.widgets.text("wheel", "", "Wheel path on a volume")
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("search_endpoint", "", "AI Search endpoint")
dbutils.widgets.text("secret_scope", "hgrag", "Secret scope")
dbutils.widgets.text("mlflow_root", "/Shared", "Workspace folder of the MLflow experiments")
dbutils.widgets.text("dataset", "", "Evaluation dataset on the volume (default eval/eval_dataset.jsonl)")
dbutils.widgets.text("k", "10", "Chunks to retrieve")
dbutils.widgets.dropdown("rerank", "true", ["true", "false"], "AI Search reranker")
dbutils.widgets.dropdown("generate", "true", ["true", "false"], "Answer and judge (Claude API calls)")

wheel_spec = dbutils.widgets.get("wheel") + "[databricks]"

# COMMAND ----------

# MAGIC %pip install "opencv-python-headless==4.11.0.86" $wheel_spec

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

# Before the first import of hgrag: the configuration is read at import.
os.environ.update(
    HGRAG_BACKEND="databricks",
    HGRAG_DBX_CATALOG=dbutils.widgets.get("catalog"),
    HGRAG_DBX_SCHEMA=dbutils.widgets.get("schema"),
    HGRAG_DBX_VOLUME=dbutils.widgets.get("volume"),
    HGRAG_DBX_SEARCH_ENDPOINT=dbutils.widgets.get("search_endpoint"),
    ANTHROPIC_API_KEY=dbutils.secrets.get(dbutils.widgets.get("secret_scope"), "ANTHROPIC_API_KEY"),
    HF_TOKEN=dbutils.secrets.get(dbutils.widgets.get("secret_scope"), "HF_TOKEN"),
    HGRAG_ASSESS_JUDGES="false",  # the production checks are not what this run measures
    MLFLOW_TRACKING_URI="databricks",
    HGRAG_MLFLOW_EXPERIMENT_ROOT=dbutils.widgets.get("mlflow_root"),
    HGRAG_MLFLOW_DIR="/tmp/hgrag_mlflow",  # only a lock file: not on a volume
)

# COMMAND ----------

from hgrag import config, evaluation, tracking
from hgrag.dbx import evaluate
from hgrag.generation import LLMConfig

generate = dbutils.widgets.get("generate") == "true"
run_key = tracking.new_run_key()
rows = evaluation.load_dataset(dbutils.widgets.get("dataset") or f"{config.dbx_volume_path()}/eval/eval_dataset.jsonl")
summary = evaluate.run_eval(
    rows,
    f"{config.dbx_volume_path()}/eval/runs/{run_key}",
    k=int(dbutils.widgets.get("k")),
    rerank=dbutils.widgets.get("rerank") == "true",
    gen_cfg=LLMConfig(provider="anthropic") if generate else None,
    judge_cfg=evaluation.judge_config() if generate else None,
    on_row=lambda r: print(r["id"], r.get("rank"), r.get("score"), (r.get("error") or "")[:120]),
    run_key=run_key,
)
print(evaluation.format_summary(summary))
print({"hgrag_eval": summary["dest"]})
