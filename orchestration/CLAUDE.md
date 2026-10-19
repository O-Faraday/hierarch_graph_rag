# orchestration/ (Airflow)

- Environnement et `uv.lock` séparés du projet principal (contraintes Airflow dans `constraints-*.txt`).
- Mis de côté pour l'instant : le runner intégré (`src/hgrag/runner.py`) fait le travail. Les DAG appellent les mêmes fonctions de `hgrag.pipeline` ; ne pas dupliquer la logique ici.
- Qdrant local : une seule tâche d'indexation pour tous les documents.
