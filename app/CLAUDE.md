# app/ (Streamlit)

- Trois pages dans `views/` : `pipeline.py`, `query.py`, `vlm.py`. Style commun dans `ui.py` (CSS dérivé de la couleur du texte : thème clair et sombre).
- Qdrant local n'accepte qu'un seul processus : la page Query se met en pause pendant un run d'indexation. Ne pas contourner.
- Les modèles lourds passent par `models.py` (`st.cache_resource`). Les clés saisies dans l'interface ne vivent que dans la session : jamais sur disque, jamais dans MLflow.
- Pas de `value=` à côté de `key=` pour un widget : initialiser `st.session_state` avec `setdefault`.
- `tests/test_app.py` rend les pages sans réseau ; le relancer après toute modification de l'interface.
