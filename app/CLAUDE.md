# app/ (Streamlit)

- Six pages dans `views/`, en deux sections : *Pipeline* (`pipeline.py`, `query.py`, `vlm.py`) et *Ragbench* (`rb_results.py`, `rb_detail.py`, `rb_summary.py`). Style commun dans `ui.py` (CSS dérivé de la couleur du texte : thème clair et sombre).
- La page Results lance le second avis (`hgrag.refine`, appels API) puis vide le cache du run (`rv.clear_cache()`).
- Les pages Ragbench partagent `ragbench_view.py` (run et question choisis : `rb_run`, `rb_qid`, `rb_order`) ; la logique (diagnostic, `review.json`) est dans `hgrag.ragbench_review`, testée sans Streamlit.
- Qdrant local n'accepte qu'un seul processus : la page Query se met en pause pendant un run d'indexation. Ne pas contourner.
- Les modèles lourds passent par `models.py` (`st.cache_resource`). Les clés saisies dans l'interface ne vivent que dans la session : jamais sur disque, jamais dans MLflow.
- Pas de `value=` à côté de `key=` pour un widget : initialiser `st.session_state` avec `setdefault`.
- `tests/test_app.py` rend les pages sans réseau ; le relancer après toute modification de l'interface.
