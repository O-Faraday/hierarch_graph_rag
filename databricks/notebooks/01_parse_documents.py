# Databricks notebook source
# MAGIC %md
# MAGIC # Parse the new PDFs of a volume with Docling
# MAGIC Thin notebook: the logic is in `hgrag.dbx.parsing`. Runtime 17.3 LTS or above.
# MAGIC OpenCV stays at 4.11: recent versions abort the process on import here (FIPS self-test of their bundled OpenSSL).

# COMMAND ----------

dbutils.widgets.text("wheel", "", "Wheel path on a volume")
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("checkpoint", "", "Checkpoint path (empty = in the volume)")
dbutils.widgets.text("secret_scope", "hgrag", "Secret scope")
dbutils.widgets.text("document_timeout", "1800", "Seconds per document (0 = no limit)")
dbutils.widgets.dropdown("retry_incomplete", "false", ["false", "true"], "Parse failed and partial documents again")

wheel_spec = dbutils.widgets.get("wheel") + "[databricks]"

# COMMAND ----------

# MAGIC %pip install "opencv-python-headless==4.11.0.86" $wheel_spec

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

# Before the first import of hgrag: the configuration is read at import.
volume_root = "/Volumes/{}/{}/{}".format(*(dbutils.widgets.get(k) for k in ("catalog", "schema", "volume")))
os.environ.update(
    HGRAG_BACKEND="databricks",
    HGRAG_DBX_CATALOG=dbutils.widgets.get("catalog"),
    HGRAG_DBX_SCHEMA=dbutils.widgets.get("schema"),
    HGRAG_DBX_VOLUME=dbutils.widgets.get("volume"),
    HGRAG_DBX_CHECKPOINT=dbutils.widgets.get("checkpoint"),
    HGRAG_DOCLING_ARTIFACTS=f"{volume_root}/models/docling",
    HGRAG_DOCLING_TIMEOUT=dbutils.widgets.get("document_timeout"),
    HF_TOKEN=dbutils.secrets.get(dbutils.widgets.get("secret_scope"), "HF_TOKEN"),
)

# COMMAND ----------

from hgrag import config
from hgrag.dbx import parsing

parsing.prefetch_models(config.DOCLING_ARTIFACTS)
parsing.discover_pdfs(spark)
parsing.parse_pending(spark)
if dbutils.widgets.get("retry_incomplete") == "true":
    parsing.retry_incomplete(spark)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {config.dbx_table(parsing.TABLE)} ORDER BY updated_at DESC"))
