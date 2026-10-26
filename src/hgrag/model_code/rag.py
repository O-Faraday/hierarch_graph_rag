"""MLflow "model from code": the question-answering application (``hgrag.mlflow_models.RagModel``)."""

import mlflow

from hgrag.mlflow_models import RagModel

mlflow.models.set_model(RagModel())
