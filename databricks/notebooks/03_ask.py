# Databricks notebook source
# MAGIC %md
# MAGIC # Ask a question: AI Search, then rebuilt sections, then a sourced answer
# MAGIC Thin notebook: the logic is in `hgrag.dbx.retrieval`. Run `01_parse_documents` and `02_index_chunks` first.
# MAGIC The answers and the judges use the Claude API, with the `ANTHROPIC_API_KEY` secret of the scope.
# MAGIC The AI Search reranker is tried first. On a workspace without it, the Hugging Face API takes over with the `HF_TOKEN`
# MAGIC secret of the scope: the question and the text of the candidate chunks are then sent there, and `hgrag_reranker` says
# MAGIC so. `rerank=false` keeps the order of the hybrid search and calls no reranker.

# COMMAND ----------

dbutils.widgets.text("wheel", "", "Wheel path on a volume")
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("search_endpoint", "databricks-structured-parsing-search", "AI Search endpoint")
dbutils.widgets.text("secret_scope", "hgrag", "Secret scope")
dbutils.widgets.text("mlflow_root", "/Shared", "Workspace folder of the MLflow experiments")
dbutils.widgets.text("question", "", "Question")
dbutils.widgets.text("k", "10", "Chunks to retrieve")
dbutils.widgets.dropdown("rerank", "true", ["true", "false"], "AI Search reranker")
dbutils.widgets.text("k_sections", "10", "Sections to read (0: the sections of the k chunks)")
dbutils.widgets.text("rerank_candidates", "50", "Ranked chunks the sections are taken from")
dbutils.widgets.dropdown("judges", "true", ["true", "false"], "LLM judges (extra API calls)")

# Installed twice: with its dependencies, then alone and forced, because pip keeps an installed wheel of the same
# version and would not take the code of a new deployment.
wheel_path = dbutils.widgets.get("wheel")
wheel_spec = wheel_path + "[databricks]"

# COMMAND ----------

# MAGIC %pip install "opencv-python-headless==4.11.0.86" $wheel_spec

# COMMAND ----------

# MAGIC %pip install --force-reinstall --no-deps $wheel_path

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
    HGRAG_K_SECTIONS=dbutils.widgets.get("k_sections"),  # hgrag.toml is not in the wheel: the project settings
    HGRAG_RERANK_CANDIDATES=dbutils.widgets.get("rerank_candidates"),
    MLFLOW_TRACKING_URI="databricks",
    HGRAG_MLFLOW_EXPERIMENT_ROOT=dbutils.widgets.get("mlflow_root"),
    HGRAG_MLFLOW_DIR="/tmp/hgrag_mlflow",  # only a lock file: not on a volume
)

# COMMAND ----------

from hgrag import tracing, tracking
from hgrag.dbx import retrieval

run_key = tracking.new_run_key()
result = retrieval.ask(
    dbutils.widgets.get("question"),
    k=int(dbutils.widgets.get("k")),
    rerank=dbutils.widgets.get("rerank") == "true",
    judges=dbutils.widgets.get("judges") == "true",
    run_key=run_key,
)
print(result["answer"] or "No passage found in the index.")
tracing.report(tracking.RETRIEVAL, run_key)

# COMMAND ----------

if result["sources"]:
    display(spark.createDataFrame(result["sources"]))
print({k: result.get(k) for k in ("model", "usage", "seconds", "dropped", "n_images", "images_missing")})
