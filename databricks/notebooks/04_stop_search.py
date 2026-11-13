# Databricks notebook source
# MAGIC %md
# MAGIC # Delete the AI Search index to stop its billing
# MAGIC Thin notebook: the logic is in `hgrag.dbx.indexing`. AI Search has no pause: an endpoint stops being billed 24 h
# MAGIC after the deletion of its last index. The endpoint, the other indexes and the `chunks` table are kept.
# MAGIC `03_ask` fails until `02_index_chunks` has created the index again, which computes the embeddings again.

# COMMAND ----------

dbutils.widgets.text("wheel", "", "Wheel path")
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("search_endpoint", "", "AI Search endpoint")

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
    HGRAG_DBX_SEARCH_ENDPOINT=dbutils.widgets.get("search_endpoint"),
)

# COMMAND ----------

from hgrag.dbx import indexing

indexing.stop_search()
