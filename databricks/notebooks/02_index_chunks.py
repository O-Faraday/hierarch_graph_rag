# Databricks notebook source
# MAGIC %md
# MAGIC # Chunk the parsed documents and index them in AI Search
# MAGIC Thin notebook: the logic is in `hgrag.dbx.indexing`. Run `01_parse_documents` first.
# MAGIC The AI Search endpoint is created if it does not exist, and it is billed until it is deleted.

# COMMAND ----------

dbutils.widgets.text("wheel", "", "Wheel path on a volume")
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("search_endpoint", "", "AI Search endpoint")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en", "Embedding endpoint")
dbutils.widgets.text("secret_scope", "hgrag", "Secret scope")
dbutils.widgets.text("mlflow_root", "/Shared", "Workspace folder of the MLflow experiments")
dbutils.widgets.dropdown("judges", "true", ["true", "false"], "LLM judges of the checks (extra API calls)")
dbutils.widgets.dropdown("reassess", "false", ["false", "true"], "Run the checks again on all the documents")
dbutils.widgets.text("test_query", "", "Question for a test search (optional)")

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
    HGRAG_DBX_EMBED_ENDPOINT=dbutils.widgets.get("embedding_endpoint"),
    ANTHROPIC_API_KEY=dbutils.secrets.get(dbutils.widgets.get("secret_scope"), "ANTHROPIC_API_KEY"),
    HGRAG_ASSESS_JUDGES=dbutils.widgets.get("judges"),
    MLFLOW_TRACKING_URI="databricks",
    HGRAG_MLFLOW_EXPERIMENT_ROOT=dbutils.widgets.get("mlflow_root"),
    HGRAG_MLFLOW_DIR="/tmp/hgrag_mlflow",  # only a lock file: not on a volume
)

# COMMAND ----------

from hgrag import config, tracing, tracking
from hgrag.dbx import indexing

run_key = tracking.new_run_key()
chunks = indexing.chunk_pending(spark, run_key=run_key)
index = indexing.sync_index(session=spark, run_key=run_key)
if dbutils.widgets.get("reassess") == "true":
    indexing.assess_documents(spark, run_key=run_key)
tracking.finish_execution(tracking.INDEXING, run_key, failed=bool(chunks["failed"]))
tracing.report(tracking.INDEXING, run_key)

# COMMAND ----------

display(spark.sql(f"""SELECT doc_id, path, count(*) AS n_chunks, max(indexed_at) AS indexed_at
    FROM {config.dbx_table(indexing.TABLE)} GROUP BY doc_id, path"""))

# COMMAND ----------

if dbutils.widgets.get("test_query"):
    display(spark.createDataFrame(indexing.search(dbutils.widgets.get("test_query"))).select(
        "path", "page_no", "section_path", "text"))

# COMMAND ----------

# Last, once the other documents are indexed: a job with failed documents must not end green.
indexing.raise_on_failures(spark, chunks)
