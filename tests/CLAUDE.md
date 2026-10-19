# tests/

- Tests hors ligne : aucun vrai modèle (Docling, embeddings, cross-encoder), aucun appel d'API. Utiliser des faux (`BagOfWordsEmbedder` dans `test_retrieval.py`, faux `_ask_image`, faux reranker dans `test_rerank.py`).
- `conftest.py` coupe le reranker et vide les préfixes d'embedding pour tous les tests ; un test qui les teste les réactive explicitement (`monkeypatch`).
- Un changement de comportement = un test qui échoue avant et passe après. Lancer `uv run pytest -q` en entier avant de committer (≈ 45 s).
- Les fichiers de contrôle (3 PDF au contenu connu) sont générés par `scripts/make_control_pdfs.py`.
