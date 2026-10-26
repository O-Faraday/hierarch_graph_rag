"""MLflow "model from code": the indexing application (``hgrag.mlflow_models.IndexingModel``)."""

import mlflow

from hgrag.mlflow_models import IndexingModel

mlflow.models.set_model(IndexingModel())
