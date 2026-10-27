# CLAUDE.md : mémoire du projet Hierarch_Graph_RAG

Lu automatiquement au début de chaque session. Les consignes ci-dessous sont durables ; le détail fonctionnel est dans `README.md`.

## Objectif

Entrepôt de connaissances RAG hiérarchique à partir de PDF :
PDF → Docling (structure hiérarchique, figures en PNG + métadonnées) → arbre des titres (chaque élément est une feuille) → chunks de paragraphes + embeddings dans **une seule collection Qdrant** → recherche (chunks → regroupement par section → reconstruction de la section) → réponse LLM sourcée.

## Langue

Répondre en **français**, de façon concise. Code, noms, commentaires et messages de commit en anglais.

## Stack

- Python ≥ 3.13, gestionnaire **`uv`** (jamais `pip install` directement dans le dépôt).
- Docling 2.x (parsing), Qdrant en mode local (un seul processus à la fois sur le dossier), Sentence-Transformers (embedding + cross-encoder), MLflow (suivi), Streamlit (app), Airflow (code dans `orchestration/`, mis de côté pour l'instant).
- Figures, tableaux et formules décrits par un modèle de vision via l'API (routeur Hugging Face) ; réponses et juge via l'API Claude.

## Structure

| Chemin | Rôle |
|---|---|
| `src/hgrag/parsing.py` | Docling + options, export des figures, manifeste (`parse_options`) |
| `src/hgrag/describe.py` | Descriptions des figures/tableaux, formules → LaTeX (modèle de vision) |
| `src/hgrag/hierarchy.py` | Arbre des titres et rattachement des éléments |
| `src/hgrag/indexing.py`, `pipeline.py` | Chunks, préfixes d'embedding, Qdrant |
| `src/hgrag/retrieval.py`, `rerank.py` | Recherche par étapes, reranking, reconstruction des sections |
| `src/hgrag/generation.py`, `evaluation.py` | Réponse LLM, juge |
| `src/hgrag/ragbench.py` | Validation sur Open RAG Bench (rapports, `questions.md`) |
| `src/hgrag/refine.py` | Second avis sur les mauvaises notes : juge de contexte (PDF entier), puis juge de génération |
| `src/hgrag/assessments/` | Contrôles de production sans référence, un module par étape (`hgrag assess`, `assessments.json`, `Feedback` MLflow) |
| `src/hgrag/tracking.py`, `tracing.py` | MLflow : runs par étape ; traces (entrées/sorties de chaque étape, parent explicite, jamais `mlflow.set_experiment`) |
| `src/hgrag/versions.py`, `registry.py`, `mlflow_models.py`, `genai_eval.py` | Versions, modèles pyfunc enregistrés (`hgrag register`), évaluation `mlflow.genai.evaluate` (`hgrag evaluate-rag`) |
| `src/hgrag/config.py` | Toute la configuration (`HGRAG_*`) |
| `src/hgrag/cli.py` | Commande `hgrag` |
| `app/` | Application Streamlit (`views/` : pipeline, query, vlm) |
| `orchestration/` | DAG Airflow |
| `tests/` | Tests hors ligne (modèles factices) |

## Commandes

```bash
uv sync                                   # installer
uv run pytest -q                          # tests (≈ 45 s, hors ligne, aucun modèle réel)
uv run streamlit run app/streamlit_app.py # application
uv run hgrag --help                       # run, query, verify, bundle, eval, describe, vlm-bench, ragbench, ragbench-report, ragbench-refine, assess, register, evaluate-rag, mlflow-ui
uv run hgrag ragbench --docs 30 --negatives 30 -k 10 --generate   # validation (tourne chez l'utilisateur, pas dans le bac à sable)
```

## Conventions

- **Lire la documentation la plus récente** de chaque bibliothèque avant de l'utiliser ou de changer son usage (signatures, options). Ne pas se fier à la mémoire.
- Toute nouvelle option va dans `config.py`, avec une variable `HGRAG_*`, une ligne dans le tableau du `README.md` et, si elle change le parsing, dans `parse_options()` (invalide le cache de parsing).
- Tout changement de comportement s'accompagne d'un test ; les tests ne chargent jamais de vrai modèle (voir `tests/conftest.py`: reranker coupé, préfixes vides) ni n'appellent d'API.
- Un changement d'embedding change la taille des vecteurs : un autre dossier/collection Qdrant (le ragbench le fait seul via `qdrant_dirname`).
- Windows est la machine de l'utilisateur : chemins `\` dans d'anciens `png_path`, fichiers en lecture seule, encodage cp1252 déjà gérés, ne pas les casser.
- Commits : un par changement logique, message en anglais, avec les lignes d'attribution demandées par la session.
- Après chaque livraison de code : relancer la suite complète, committer, puis fournir le zip du dépôt (`git archive`).

## Interdits et vigilance

- **Jamais de clé d'API écrite sur disque, dans MLflow, dans un export ou un zip.** Clés uniquement via l'environnement / `.env` (`ANTHROPIC_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`). `.env` reste hors de git et hors des zips. Toute nouvelle variable `HGRAG_*` va aussi dans `env.example` (modèle commenté du `.env`, sans valeur secrète ; un test vérifie qu'il ne nomme que des variables lues par le code).
- Ne pas lancer Docling CodeFormulaV2 sur CPU par défaut (≈ 3 min par élément, boucles) : `HGRAG_CODE_ENRICHMENT=0`, formules par le modèle de vision (`HGRAG_FORMULA_MODE=vlm`).
- Ne pas préfixer le texte vectorisé avec le titre du document (éléments `title` de Docling) : les chunks de page de garde gagnent toutes les requêtes. Ne pas indexer les références ni les remerciements.
- Le bac à sable ne peut pas exécuter Docling réel, HF, Anthropic ni les vrais embeddings : ne jamais annoncer un gain de qualité non mesuré ; les effets se mesurent sur les runs de l'utilisateur (`summary.json`, `questions.md`).
- Open RAG Bench est sous licence CC BY-NC 4.0 : outil de validation interne, pas de données client.
- Les métriques du ragbench sont maison (recouvrement 5-grammes ou titre identique, juge 1-5) : ne pas les présenter comme les métriques officielles de Vectara.
- Les fichiers téléversés par l'utilisateur sont des originaux : travailler sur une copie.

## Ne pas oublier quand on analyse un run

Regarder dans l'ordre : `summary.json` (doc hit, section hit@1/@10, MRR, context recall, note), puis `questions.md` (trié par score croissant), puis `results.jsonl` pour séparer échec de recherche (section absente) et échec de génération (section présente, mauvaise réponse). Les questions sur figures sans valeurs écrites et les questions très générales sont les faiblesses connues.
