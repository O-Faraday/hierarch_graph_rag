#!/usr/bin/env bash
# Start Airflow locally (api-server, scheduler, dag-processor, triggerer) with LocalExecutor.
set -euo pipefail
cd "$(dirname "$0")"

# Project .env (HF_TOKEN, HGRAG_* ...) is exported to Airflow and inherited by the tasks.
if [ -f ../.env ]; then set -a; . ../.env; set +a; fi

export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
# LocalExecutor spawns `parallelism` worker processes at startup (default 32): keep it small on a laptop.
export AIRFLOW__CORE__PARALLELISM="${AIRFLOW__CORE__PARALLELISM:-4}"
# Change the API port if 8080 is taken (also set HGRAG_AIRFLOW_URL for the Streamlit app).
export AIRFLOW__API__PORT="${AIRFLOW__API__PORT:-8080}"
# Interpreter of the project venv (Docling, sentence-transformers, Qdrant) used by external_python tasks.
export HGRAG_PYTHON="${HGRAG_PYTHON:-$PWD/../.venv/bin/python}"

exec uv run airflow standalone
