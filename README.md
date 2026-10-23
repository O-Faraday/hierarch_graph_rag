# Hierarchical RAG à partir de PDF

PDF → Docling (structure hiérarchique, figures en PNG, descriptions par VLM) → arbre des titres
(éléments en feuilles) → chunks des éléments textuels (paragraphes, listes, notes, formules, code, tableaux) + embeddings anglais dans **une seule** collection
Qdrant. Un second flux (retrieval) retrouve les chunks proches d'une requête et reconstruit le
document à partir de **tous** les éléments de la section. Interface : Streamlit (lance les étapes et
affiche le graphe des tâches). Suivi : MLflow. Environnements : uv. Airflow (orchestration) viendra
se brancher sur les mêmes étapes une fois les workflows stabilisés : voir §5.

```
PDF ─► parse ─► hierarchy ─► index (Qdrant)          pipeline d'indexation
         │          │             ▲
   document.json  hierarchy.json  └── chunks (éléments textuels) + embeddings

requête ─► embed ─► Qdrant ─► regroupement par section ─► reconstruction (parallèle)   pipeline de retrieval
```

---

## 1. Protocole de lancement (WSL / Linux : app + MLflow, sans Airflow)

Tout se passe dans WSL (Ubuntu), dans `~/hierarch_graph_rag`. Aucun Airflow à démarrer : l'app lance
elle-même les étapes dans un sous-processus.

### 1.1 Installation (une fois)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd ~/hierarch_graph_rag
uv sync                      # un seul environnement (racine). `orchestration/` n'est pas nécessaire.
```

Créer un fichier `.env` à la racine (jamais commité ; lu par l'app, la CLI et le runner : `HF_HUB_OFFLINE=1`, `ANTHROPIC_API_KEY=…`, `HF_TOKEN=…`, les variables `HGRAG_*`) :

```ini
HF_TOKEN=hf_xxx                       # évite les limites de téléchargement Hugging Face
HF_HUB_OFFLINE=1                      # une fois les modèles en cache : plus aucun appel à Hugging Face
ANTHROPIC_API_KEY=sk-ant-xxx          # réponse générée (page Query) et juge de `hgrag eval`
```

Déposer les PDF dans `data/pdf_sample/`. Vérifier sans télécharger de modèle : `uv run pytest`.

### 1.2 Lancement : 2 terminaux (VS Code : *Terminal → New Terminal*)

| Terminal | Commande | Adresse |
|---|---|---|
| **1 · MLflow** | `uv run hgrag mlflow-ui` | http://localhost:5000 (prêt après ~30 s) |
| **2 · App** | `uv run streamlit run app/streamlit_app.py` | http://localhost:8501 |

Si le port 5000 est pris : `uv run mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5001`.

1. Page **Pipelines** :
   - Le tableau *Documents* montre ce qui existe déjà sur disque (parsing, hiérarchie, figures PNG).
   - **Run whole pipeline** : parse → hierarchy → index pour tous les PDF.
   - **Rebuild hierarchy + index** : réutilise le parsing Docling déjà sur disque (pas de re-parse, donc
     rapide), reconstruit l'arbre des titres et les chunks.
   - **Re-index only** : re-découpe et re-embed depuis le parsing existant (changement de modèle, de
     taille de chunk, de `HGRAG_MIN_CHUNK_WORDS`…).
   - *Custom run* : choisir les documents et les étapes à relancer.
   - Dans un run terminé : **Retry failed** (relance les PDF en échec depuis la première étape en
     échec) et **Re-run selected rows** (cocher des lignes du tableau des instances).
   - Le tableau « documents × étapes » se colore au fil de l'exécution (bleu = en cours, vert = réussi, rouge pastel = échec, violet = réutilisé depuis le disque) (étapes par PDF :
     `[réussies/total]` ; étapes non sélectionnées : *skipped*), avec durées, sorties, erreurs, log et
     étapes MLflow. **Cancel** arrête le run. Réindexer un document remplace ses chunks (idempotent).
2. Page **Query** : une fois l'indexation terminée, saisir une requête ; la sortie de chaque étape
   s'affiche au fur et à mesure (embedding, chunks Qdrant, regroupement, sections reconstruites).
3. MLflow : expériences `hgrag_pipeline` et `hgrag_retrieval` (un run parent par exécution, un run
   enfant par étape).

Arrêt : `Ctrl+C` dans chaque terminal. Un run d'indexation continue même si l'on ferme l'onglet.

### 1.3 Sans l'interface (CLI)

```bash
uv run hgrag run                       # parse → hiérarchie → index (--no-vlm pour sauter le VLM)
uv run hgrag query "three new branches" -k 5          # --children, --json
uv run python -m hgrag.runner indexing --no-vlm       # même exécution que le bouton de l'app (statut + MLflow)
uv run python -m hgrag.runner indexing --steps hierarchy,index --pdf rapport.pdf   # relance ciblée
```

`--no-mlflow` désactive le suivi MLflow sur `run` et `query`.

### 1.3 ter Benchmark rapide et juge LLM (ligne de commande)

**Pourquoi Docling n'entre pas dans le benchmark.** Le parsing est la seule étape lente et sa sortie
(`document.json`) ne dépend de rien en aval : hiérarchie, chunks, embeddings, recherche et génération partent
tous de ce fichier. Les sorties Docling des 3 PDF de contrôle sont donc **figées** dans
`data/pdf_control/parsed/` (celles de ta machine, avec les descriptions si `hgrag describe` a été lancé avant) et le benchmark reconstruit
tout le reste en quelques secondes, dans un store privé (`data/eval/store`, celui de l'app n'est pas touché).
Pour changer un réglage de chunking (`HGRAG_CHUNK_LABELS`, `HGRAG_MIN_CHUNK_WORDS`, modèle d'embedding) et
mesurer l'effet : relancer `eval`, une minute. Après un nouveau parse réel : `uv run hgrag freeze`.

```bash
uv run hgrag eval --no-generate        # rang de la bonne section (hit@1, hit@k, MRR), aucun appel API
uv run hgrag eval                      # + réponse générée par Claude + note du juge (ANTHROPIC_API_KEY du .env)
uv run hgrag eval --ids q01,q08 -k 3   # quelques questions
uv run hgrag eval --kind figure,list   # par type d'élément
uv run hgrag eval --live               # évaluer le store de l'app (data/output + data/qdrant)
```

Le **juge** (`claude-haiku-5-5`, `HGRAG_JUDGE_MODEL`) compare chaque réponse à la vérité terrain du dataset et
renvoie une **note de 1 à 5** et une **justification** (5 = complet et exact, 3 = partiel, 1 = faux ou inventé ;
pour les questions sans réponse, s'abstenir = 5, inventer = 1). Sortie : une ligne par question dans le terminal,
puis `data/eval/<run>/` (`results.jsonl`, `summary.json`, `report.md`) et l'expérience MLflow `hgrag_eval`.

### 1.3 quater Comparer des VLM pour décrire les figures (API, rien d'installé)

Les modèles tournent chez des fournisseurs via le **router Hugging Face** (`HF_TOKEN` dans `.env`, jeton avec la
permission « Make calls to Inference Providers »; facturation à l'appel, quelques centimes ici) ou via l'API Claude.
Chaque modèle décrit les 3 figures de contrôle; le juge Claude vérifie la description contre une liste de faits
(`data/pdf_control/figure_truth.json` : type, titre, axes, valeurs, tendance) et relève ce qui est inventé.

```powershell
uv run hgrag vlm-bench --list-models          # modèles à image servis par le router
uv run hgrag vlm-bench                        # candidats par défaut
uv run hgrag vlm-bench --models hf:google/gemma-3-12b-it anthropic:claude-haiku-5-5 --figures botany_f1
```

Résultats dans `data/eval/vlm/<date>/` ; lecture interactive dans la page **VLM comparison** de l'app (classement,
image à côté des descriptions, faits trouvés/ratés, inventions). Granite-Vision, SmolVLM et Qwen2.5-VL-3B ne sont
pas servis par le router au moment de l'écriture : ils ne sont pas comparables par API.

### 1.3 quinquies Descriptions des figures et des tableaux (API, rien d'installé)

Après le parsing Docling (qui n'utilise plus SmolVLM), deux appels API enrichissent l'indexation. Il faut `HF_TOKEN` dans `.env` ;
sans jeton, le parsing reste valide et l'étape est simplement sautée (avertissement dans le log et le manifest).

- **Figures** : chaque PNG est décrit par `HGRAG_FIGURE_VLM` (défaut `hf:Qwen/Qwen3-VL-30B-A3B-Instruct`, gagnant de
  `hgrag vlm-bench` : 5/5, tous les faits, aucune invention). La description (`figures.json`) devient **un chunk à part**
  (légende + description) donc la figure est trouvable par sa recherche ; le PNG reste envoyé au LLM de réponse.
- **Tableaux** : le Markdown extrait par Docling (donc les chiffres exacts) est envoyé en **texte** à `HGRAG_TABLE_MODEL`
  (défaut = le modèle des figures) ; la description (`tables.json`) est **ajoutée au texte d'embedding** des chunks du
  tableau (placée avant les lignes), le texte du chunk reste les lignes brutes.

```powershell
uv run hgrag describe            # figures + tableaux des documents déjà parsés (sans relancer Docling)
uv run hgrag describe --force    # refaire les descriptions existantes
```

Puis **Rebuild hierarchy + index** dans l'app (ou `uv run hgrag run` en cas de nouveau parsing) pour que les descriptions
soient indexées. Un tableau inchangé n'est pas redécrit (clé = hash du Markdown). Dans l'app : case « Describe figures and
tables after parsing » et bouton « Describe figures and tables of the parsed documents ».

### 1.3 sexies Valider sur Open RAG Bench (PDF arXiv réels, tableaux et figures)

[Open RAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) (Vectara, CC BY-NC 4.0) : 1 000 articles arXiv,
3 045 questions (texte seul, texte+image, texte+tableau, texte+tableau+image), la vérité terrain est une **section** du papier.
La taille de l'échantillon est configurable :

```powershell
uv run hgrag ragbench --docs 10                                  # ~30 questions, premier essai
uv run hgrag ragbench --docs 100 --negatives 20                  # + 20 papiers sans question (distracteurs)
uv run hgrag ragbench --docs 30 --sources text-table,text-table-image   # seulement les questions avec tableaux
uv run hgrag ragbench --docs 10 --generate                       # + réponses Claude jugées (descriptions HF actives si HF_TOKEN)
uv run hgrag ragbench --docs 10 --no-describe                    # sans descriptions de figures/tableaux
uv run hgrag ragbench --docs 100 --prepare-only                  # télécharger/parser/indexer sans évaluer (à laisser tourner)
uv run hgrag ragbench --docs 30 --negatives 30 -k 10 --generate --gen-model <modèle> \
    --reference-run data/ragbench/runs/<run>                     # autre modèle de génération, même juge, contexte vérifié identique
uv run hgrag ragbench --docs 30 --negatives 30 -k 10 --generate --questions ee9ea96b,1ccbaf52   # seulement ces questions (ids ou préfixes)
```

Étapes : téléchargement des PDF (arXiv, 1 requête / 3 s, `data/ragbench/pdfs`), parsing Docling **mis en cache** par document
(`data/ragbench/output` : un second run ou un échantillon plus grand ne reparse pas ce qui l'est déjà), index dans un store
privé par échantillon (`data/ragbench/<n>_<seed>...`, jamais celui de l'app). Le même `--seed` donne le même échantillon.
Un PDF qui ne se télécharge ou ne se parse pas est ignoré (listé dans `summary.json`).

Les sections de référence viennent du parsing de Vectara, pas du nôtre : on ne compare donc **jamais par index** mais par
recouvrement de texte (5-grammes de mots) :

| Métrique | Sens |
|---|---|
| `doc hit@1/@k` | le bon article est dans les sections retournées |
| `section hit@1/@k`, MRR | un chunk retourné de la section est dans le texte de référence (≥ 50 % de ses 5-grammes) |
| `context recall` | part du texte de référence présente dans les sections **reconstruites** : l'apport de la hiérarchie |
| `mean_score` (avec `--generate`) | note 1-5 du juge Claude de la réponse contre la réponse de référence |

Détail par type de question (`text`, `text-image`, `text-table`...) et par style (`extractive`/`abstractive`). Résultats dans
`data/ragbench/runs/<date>/` (`results.jsonl`, `summary.json`, `report.md`, `questions.md`) et dans MLflow (expérience `hgrag_ragbench`).

Une section récupérée est « sur la référence » si un de ses chunks recouvre au moins 50 % des 5-grammes de la référence **ou** si son titre est celui de la section de référence (le LaTeX écrit différemment par les deux parseurs fait chuter le recouvrement) ; `on_gold_by` indique `text` ou `title`.

`questions.md` regroupe dans un seul document, par score croissant (note du juge, puis recall), chaque question avec la réponse attendue,
la réponse générée, le commentaire du juge, un extrait de la section de référence, les 3 premières sections récupérées et le chemin du PDF
(`data/ragbench/pdfs/<id arXiv>.pdf`). Pour un run existant : `uv run hgrag ragbench-report data/ragbench/runs/<run>`.

Références litigieuses : `data/ragbench_disputed.json` liste les questions dont la réponse de référence a été vérifiée fausse
dans le PDF (`{id: raison}`). Elles restent dans les métriques brutes ; `summary.json` donne en plus le bloc
`excluding_disputed` (mêmes métriques sans elles) et `questions.md` les marque. `ragbench-report` recalcule aussi ce bloc
pour un run existant.

**Second avis sur les mauvaises notes** (`uv run hgrag ragbench-refine data/ragbench/runs/<run>`, ou le bouton
*Run the judges* de la page *Results*). Le premier juge ne voit que la question, la référence et la réponse : il ne
distingue ni un échec de recherche d'un échec de génération, ni une référence fausse d'une réponse fausse. Pour chaque
question notée sous le seuil (`--threshold`, défaut 4, donc de 1 à 3) :

1. un **juge de contexte** lit le PDF entier et le contexte reçu par le générateur (sections et figures, reconstruites
   depuis le run, sans rien régénérer) et donne `context_score` (1-5) ;
2. si `context_score ≥ HGRAG_REFINE_CONTEXT_OK` (4), un **juge de génération** renote la réponse avec le PDF et le
   contexte sous les yeux (tolérance sur les valeurs lues sur un graphique, le PDF prime sur une référence imprécise) :
   `refine_score` et `refine_cause` (`reference`, `judge`, `generation`, `none`) ; sinon `refine_score` garde la note
   et la cause est `context`.

`score` n'est jamais écrasé : `summary.json` ajoute `mean_refined_score`, `pct_refined_4_plus` et `refine_causes`, et
`questions.md` montre les deux avis. Modèle : `claude-haiku-5-5` (`HGRAG_REFINE_MODEL`), comme la génération et le premier juge (repli serveur en cas de
refus seulement avec Sonnet ou Opus) ; le PDF, les figures et le contexte sont mis en cache entre les deux appels d'une question. Coût : le PDF entier
est envoyé (≈ 30 à 60 k tokens par question). `results.jsonl` est réécrit après chaque question : un run interrompu
reprend où il s'est arrêté (`--force` pour tout refaire). Cette note affinée reste une mesure maison.

**Analyse des runs dans l'app** (section *Ragbench* de `uv run streamlit run app/streamlit_app.py`) :

| Page | Contenu |
|---|---|
| Results | choix du bench (`config.key`, ex. `n30_s0_neg30`) puis du run (y compris ceux d'un autre dossier de travail, `data/ragbench/<nom>/runs`, préfixés de son nom) ; chiffres clés ; second avis sur les notes sous un seuil (bouton *Run the judges*) ; tableau triable et filtrable (source, type, diagnostic, note, gardée) ; une ligne sélectionnée ouvre son détail |
| Question detail | note, rangs, réponses attendue et générée, commentaire du juge, diagnostic par règles et piste de correctif, section de référence et sections retrouvées ; case « Keep in the bench » et note |
| Summary | bench complet contre sélection gardée (global, par source, par type), sur la note corrigée là où le second avis existe, la note du premier juge restant affichée en référence ; notes du premier juge contre second avis (moyennes, matrice note → note affinée, causes, détail par question) ; répartition des rangs, exclusions, pistes d'amélioration classées par effectif |
| Compare runs | un run de référence (par défaut le `--reference-run` du nouveau) et un nouveau run du même bench : réglages qui diffèrent, métriques côte à côte avec l'écart (premier juge et notes corrigées), questions corrigées / en recul / meilleures / moins bonnes, sections gagnées ou perdues ; une ligne ouvre la question dans le nouveau run |

Le diagnostic est fait par règles, hors ligne (`src/hgrag/ragbench_review.py`) : erreur, référence litigieuse, correct,
papier absent, section absente, refus, lecture de figure, contradiction, section mal classée, réponse partielle. Les
décisions « garder / exclure » et les notes sont propres au run (`<run>/review.json`) : elles ne changent ni
`summary.json` ni les autres runs. Une question de `data/ragbench_disputed.json` part exclue.

### Niveaux de titres (chiffres romains) et légendes

- Docling donne parfois de mauvais niveaux aux titres des articles scientifiques (`V. CONCLUSION` et `REFERENCES` rangés sous
  `IV. DISCUSSION`). La hiérarchie les corrige (`hierarchy.heading_levels`) : une suite `I. II. III. IV. V.` est lue comme les
  parties de premier niveau (donc `V.` n'est pas la lettre V après `D.`), les lettres `A. B. C.` qui suivent en sont les
  sous-sections, et References / Acknowledgments / Appendix / Supplementary materials reviennent au premier niveau dès que le
  document a plusieurs titres de premier niveau. Le journal indique combien de titres ont été corrigés ; `HGRAG_FIX_HEADING_LEVELS=0`
  désactive la correction.
- Les **légendes** sont indexées (`caption` dans `HGRAG_CHUNK_LABELS`) et rattachées à la section de leur figure ou tableau
  (avant, un chunk de légende n'avait pas de section et ne pouvait pas être retrouvé) ; la légende d'un tableau est affichée avant
  ses lignes dans la section reconstruite.

### 1.3 bis Tracer et contrôler avec les 3 PDF de contrôle

Trois PDF au contenu **connu** sont fournis dans `data/pdf_control/` (régénérables avec
`uv run python scripts/make_control_pdfs.py`, déterministes) avec `expected.json` qui décrit ce que le
pipeline doit retrouver :

| PDF | Ce qu'il contrôle |
|---|---|
| `01_aurora_bank_annual_report.pdf` | 2 niveaux de titres, 1 tableau, 1 figure |
| `02_alpine_botany_field_guide.pdf` | 3 niveaux de titres, 2 figures |
| `03_network_operations_handbook.pdf` | liste à puces, tableau, bloc de code, 0 figure |

Chaque PDF contient aussi des phrases « aiguilles » (ex. *« three new branches in the Alsace region »*) :
la requête correspondante doit ressortir en tête la bonne section.

1. Dans l'app (page *Pipelines*) : **Use the 3 control PDFs** → **Run whole pipeline**.
2. Puis ouvrir **Control check** → **Run the check** (ou en terminal, app arrêtée : `uv run hgrag verify`).
   Le rapport liste chaque contrôle : fichiers de parsing, figures PNG, tableaux, titres et niveaux, imbrication
   de l'arbre, chunks indexés, requêtes → section attendue. Code retour 1 s'il y a un échec.

**Jeu d'évaluation** : `data/pdf_control/eval_dataset.jsonl` (26 questions, une par ligne : `question`,
`expected_answer`, `pdf`, `section` attendue, `source_kind` = paragraph / table / list / code / figure / derived /
unanswerable, `evidence` = extraits textuels vérifiables dans le PDF, `requires_vlm`). Les questions `figure`
n'ont leur réponse que dans l'image (retrouvables après `hgrag describe` + réindexation) ; les 2 `unanswerable` testent
l'abstention. `tests/test_eval_dataset.py` vérifie que chaque réponse est étayée par le PDF.

**Tout récupérer pour analyse** : page *Pipelines* → **Export for analysis** → **Prepare the analysis zip** →
**Download** (ou `uv run hgrag bundle`, zip écrit dans `data/bundles/`). Il contient : `output/` (Docling JSON et
Markdown, figures PNG, arbre, chunks de chaque document), `runs/` (status et log des 10 derniers runs), `mlflow/mlflow.db`,
`control/` (attendus, jeu d'évaluation, contrôle structurel), `env/uv.lock` et `bundle_info.json` (versions des
bibliothèques, configuration, éléments absents).

Traces laissées par chaque étape (dans MLflow, par document, et sur disque dans `data/output/<doc_id>/`) :

| Étape | Métriques MLflow | Artefacts |
|---|---|---|
| parse | pages, figures (PNG, décrites), tableaux, titres, paragraphes, items par label Docling, durée de conversion | `manifest.json`, `figures.json`, `document.md`, PNG des figures |
| hierarchy | sections par niveau, profondeur, feuilles, sections vides, éléments par type | `hierarchy.json`, `hierarchy_tree.txt` (arbre indenté avec les éléments de chaque section), `tree_stats.json` |
| index | chunks par document, chunks écartés (par label), mots par chunk (moy./max), sections couvertes, dimension des vecteurs, points dans Qdrant, durées (modèle / découpage / embeddings) | `chunks/chunks.jsonl`, `index_summary.json` |

Dans l'app, **Inspect a document** montre pour un PDF : stats de parsing, arbre des titres, chunks, figures
avec leurs métadonnées. Les logs du run (`Log`) reprennent les mêmes chiffres.

### 1.4 Machine limitée

- Le parsing Docling + VLM est le poste le plus lourd : décocher le VLM (ou `--no-vlm`) pour l'alléger.
- Plafonner WSL2 dans `C:\Users\<vous>\.wslconfig`, puis `wsl --shutdown` :
  ```ini
  [wsl2]
  memory=8GB
  processors=4
  ```

### 1.5 Règles à respecter

- **Qdrant en mode local = un seul processus.** L'app suspend la page *Query* pendant une
  indexation et refuse un second run simultané. Ne pas lancer `hgrag query` en parallèle d'un run.
  Pour des accès concurrents : serveur Qdrant, puis adapter `open_store`.
- Lancer les commandes **depuis la racine** du projet.
- Garder le projet dans le système de fichiers WSL, pas sous `/mnt/c/...` (très lent). Ne jamais
  copier un `.venv` d'une machine à l'autre : le supprimer et relancer `uv sync`.
- Les modèles locaux (Docling, embedding bge, reranker) sont téléchargés au premier usage : le premier run est long.

---

## 2. Dépannage

| Symptôme | Cause / remède |
|---|---|
| `uv: command not found` | `source ~/.local/bin/env` (ou rouvrir le terminal) |
| `mlflow-ui` « sans effet » | normal : le serveur met ~30 s à démarrer et occupe le terminal. Ouvrir http://localhost:5000 |
| Qdrant « already accessed by another instance » | un autre processus tient le store (run d'indexation ou `hgrag query` en cours) |
| Run « failed — the runner process died » | voir le log du run (onglet *Log* de la page Pipelines) ; souvent mémoire insuffisante : décocher le VLM |
| `0 chunks` à l'indexation | voir les logs `dropped ... by label` ; seuls les éléments de `HGRAG_CHUNK_LABELS` sont indexés (figures exclues), filtre `HGRAG_MIN_CHUNK_WORDS` (défaut 3, `0` = off) |
| `No Qdrant store at ...` | indexer d'abord |
| `No space left on device` (WSL) | vous êtes dans la distro `docker-desktop` : utiliser `wsl -d Ubuntu` |

---

## 3. Structure du dépôt

| Chemin | Rôle |
|---|---|
| `src/hgrag/parsing.py` | Étape 1 : Docling, export PNG des figures, métadonnées des figures |
| `src/hgrag/hierarchy.py` | Étape 2 : arbre des titres, éléments rattachés aux sections |
| `src/hgrag/indexing.py` | Étapes 3-4 : chunks des éléments textuels, embeddings, Qdrant |
| `src/hgrag/retrieval.py` | Retrieval : requête → chunks proches → sections reconstruites |
| `src/hgrag/tracking.py` | Suivi MLflow : un run parent par exécution, un run enfant par étape |
| `src/hgrag/pipeline.py`, `cli.py` | Étapes en fonctions, CLI (`run`, `query`, `mlflow-ui`) |
| `src/hgrag/runner.py` | Exécuteur local (sous-processus) : graphe des tâches, `status.json`, log, annulation |
| `scripts/make_control_pdfs.py`, `data/pdf_control/` | Les 3 PDF de contrôle et `expected.json` |
| `src/hgrag/verify.py` | `hgrag verify` : compare la sortie du pipeline à `expected.json` |
| `app/` | Streamlit : page *Pipelines* (lancement, graphe, statuts) et page *Query* |
| `orchestration/` | **Pour plus tard** : projet uv séparé, Airflow 3.3.2 + DAGs `hgrag_pipeline` / `hgrag_retrieval` |
| `data/runs/<run_key>/` | `status.json` et `log.txt` de chaque run de l'app |
| `data/pdf_sample/` | PDF d'entrée |
| `data/output/<doc_id>/` | `document.json`, `figures/*.png`, `figures.json`, `hierarchy.json`, `manifest.json` |
| `data/qdrant/` | Stockage Qdrant local (collection unique `hgrag_chunks`) |
| `data/mlflow/` | Base SQLite et artefacts MLflow |

**Pourquoi deux projets uv (Airflow, plus tard).** Le fichier de contraintes d'Airflow épingle `tokenizers`,
`huggingface_hub`, `qdrant-client`… en conflit avec Docling / sentence-transformers. Airflow vit
donc dans `orchestration/` et appelle le pipeline via `@task.external_python` sur le `.venv` racine
(variable `HGRAG_PYTHON`, par défaut `<racine>/.venv/bin/python`).

---

## 4. Détails de fonctionnement

### Parsing et hiérarchie
Docling avec `HeadingHierarchyOptions(enabled=True)`, figures exportées en PNG
(`figures/figure-NNN.png`) avec leurs métadonnées (`figures.json` : référence, parent, page,
bbox, légende, dimensions/dpi, chemin PNG, description VLM, classification). `hierarchy.json` :
arbre des titres, chaque section contient ses éléments (paragraphe, liste, tableau en Markdown,
figure, formule, code, note…).

### Indexation
Seuls les **paragraphes** sont découpés (HybridChunker, tokenizer aligné sur le modèle
d'embedding, au plus `HGRAG_CHUNK_MAX_TOKENS` = 256 tokens) puis embeddés (`BAAI/bge-base-en-v1.5` par défaut, 768 dimensions, cosinus ;
les préfixes de requête/passage du modèle sont appliqués automatiquement). Les chunks de moins
de 3 mots sont écartés. Chaque point Qdrant porte `doc_id`, `section_ref`, `section_path`,
`page_no`, `text`… ; ids déterministes, ré-indexation idempotente par document.

### Retrieval
1. La requête est embeddée avec le même modèle (préfixe de requête compris), les `HGRAG_RERANK_CANDIDATES` chunks les plus proches sont lus. **Recherche hybride** : chaque chunk porte aussi un vecteur creux BM25 (mots-clés, IDF calculé par Qdrant) ; les deux listes de candidats sont fusionnées avec `HGRAG_KEYWORD_WEIGHT` (30 % mots-clés, 70 % dense, scores normalisés min-max), ce qui fait remonter les termes rares (noms de lois, symboles) qu'un embedding dense dilue. Une collection indexée avant cet ajout reste en recherche dense seule ; un cross-encoder les reclasse et garde les `k` meilleurs (`--no-rerank` / `HGRAG_RERANK=0` pour s'en passer).
2. Regroupement par section (`doc_id` + `section_ref`), meilleur score par section.
3. Chaque section est reconstruite depuis `hierarchy.json` avec **tous** ses éléments (paragraphes,
   listes, tableaux, figures avec légende + description VLM + PNG, notes), en parallèle (pool de
   threads ; une tâche mappée par section dans le DAG Airflow futur).

4. **Réponse (étape 5 de la page Query)** : les sections reconstruites, numérotées `[1]`, `[2]`…, deviennent les
   sources d'un prompt envoyé à un LLM en streaming. Le modèle doit citer ses sources, répondre « je ne peux pas
   répondre à partir des documents » si l'information n'y est pas, et ne pas reprendre de chiffres des
   descriptions de figures (marquées *machine-generated, unverified*). Deux fournisseurs : **Claude API**
   (`ANTHROPIC_API_KEY`, modèle par défaut `claude-haiku-5-5`) ou un serveur local **compatible OpenAI**
   (Ollama : `http://localhost:11434/v1`, modèle au choix). Réglages dans l'expander *Answer generation*
   (la clé saisie n'est gardée que pour la session). Si les sections dépassent le budget
   `HGRAG_LLM_MAX_CONTEXT_CHARS`, les moins bien classées sont écartées (signalé). L'étape `generate` est tracée
   dans MLflow (`hgrag_retrieval` : tokens, premier token, `prompt.txt`, `answer.txt`).

Sont embeddés les éléments textuels listés dans `HGRAG_CHUNK_LABELS` (par défaut : paragraphes, listes, notes, formules, code, tableaux, légendes ; chaque chunk reste rattaché à sa section). Les figures sont **décrites par un modèle via API** (voir 1.3 quinquies) : la description devient un chunk ; leur PNG est aussi joint à la section et envoyé au LLM de réponse. Après un changement de `HGRAG_CHUNK_LABELS`, relancer **Re-index only** dans l'app.

### Application Streamlit
- **Pipelines** : lance l'indexation dans un sous-processus (`hgrag.runner`), graphe des tâches
  coloré par état (tâches par PDF : `[réussies/total]`), tableau des instances (durée, sorties,
  erreurs), log, historique des runs et étapes MLflow du run. Un PDF en échec n'arrête pas les autres.
  L'état est lu dans `data/runs/<run_key>/status.json` : fermer l'onglet n'interrompt pas le run.
- **Query** : embedding, chunks bruts, regroupement, sections reconstruites (Markdown + figures).
  Suspendue pendant une indexation (Qdrant local = un seul processus).

### MLflow
Expériences `hgrag_pipeline` (étapes `parse`, `hierarchy`, `index`) et `hgrag_retrieval`
(`search`, `reconstruct`). Run parent nommé d'après un identifiant d'exécution (`run_key`, généré par l'app / la CLI ;
le `run_id` Airflow plus tard), un run enfant par étape ou élément mappé (PDF, section), avec paramètres, métriques
(durées, nombres de chunks / sections / éléments, scores) et artefacts. Une erreur MLflow ne fait
jamais échouer un pipeline. Serveur distant : `MLFLOW_TRACKING_URI`.

### Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | Token Hugging Face |
| `HGRAG_PDF_DIR` | `data/pdf_sample` | Dossier des PDF |
| `HGRAG_OUTPUT_DIR` | `data/output` | Sorties du parsing |
| `HGRAG_QDRANT_PATH` | `data/qdrant` | Stockage Qdrant |
| `HGRAG_COLLECTION` | `hgrag_chunks` | Nom de la collection |
| `HGRAG_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Modèle d'embedding. Changer de modèle change la taille des vecteurs : réindexer dans un autre dossier/collection (le ragbench le fait seul, `data/ragbench/<échantillon>/qdrant-<modèle>`) ; `all-MiniLM-L6-v2` reste utilisable |
| `HGRAG_EMBED_QUERY_PREFIX` / `HGRAG_EMBED_DOC_PREFIX` | selon le modèle | Préfixes d'entraînement (bge : instruction sur la requête ; e5 : `query: ` / `passage: ` ; Qwen3-Embedding : `Instruct: <tâche>\nQuery:`). `""` pour les désactiver |
| `HGRAG_QWEN3_TASK` | `Given a question about a scientific or technical document, retrieve the passages that answer it` | Instruction de tâche de `Qwen/Qwen3-Embedding-0.6B` (en anglais, comme le conseille la fiche du modèle) |
| `HGRAG_KEYWORD_WEIGHT` | `0.3` | Part du score mots-clés (BM25) dans le classement des candidats ; `0` = dense seul |
| `HGRAG_CHUNK_MAX_TOKENS` | `256` | Taille maximale des chunks (jamais au-delà de la fenêtre du modèle) |
| `HGRAG_RERANK` | `1` | Reclassement par cross-encoder : la recherche vectorielle lit `HGRAG_RERANK_CANDIDATES` chunks, le cross-encoder garde les `k` meilleurs |
| `HGRAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L12-v2` | Cross-encoder (rapide sur CPU : ~4 s par question). Plus lourds : `BAAI/bge-reranker-v2-m3` (~26 s par question sur CPU) ; `hf:<modèle>` appelle le routeur Hugging Face (tâche text-ranking, `HF_TOKEN`), par exemple `hf:Qwen/Qwen3-Reranker-0.6B`, trop lent en local (~5 min par question) |
| `HGRAG_RERANK_CANDIDATES` / `HGRAG_RERANK_MAX_LENGTH` | `30` / `512` | Candidats reclassés / tokens lus par paire |
| `HGRAG_MIN_CHUNK_WORDS` | `3` | Taille minimale d'un chunk (`0` = off) |
| `HGRAG_CHUNK_LABELS` | `text,paragraph,list_item,footnote,formula,code,table,caption` | Éléments Docling qui deviennent des chunks (retirer `table`, `code`… pour restreindre) |
| `HGRAG_DESCRIBE_PICTURES` | `1` si `HF_TOKEN` est défini | Décrire figures et tableaux après le parsing (API) |
| `HGRAG_FIGURE_VLM` | `hf:Qwen/Qwen3-VL-30B-A3B-Instruct` | Modèle de description des figures |
| `HGRAG_TABLE_MODEL` | = `HGRAG_FIGURE_VLM` | Modèle de description des tableaux (texte) |
| `HGRAG_MIN_PROSE_WORDS` | `5` | Les chunks de texte de moins de N mots (ligne de date, nom d'auteur…) ne sont pas indexés |
| `HGRAG_HEADING_CONTEXT` | `99` | Nombre de titres (les plus proches) ajoutés devant le texte embeddé ; `0` = aucun. Le titre du document n'est jamais ajouté |
| `HGRAG_EXCLUDED_SECTIONS` | `references\|bibliography\|acknowledg…` | Expression régulière des sections non indexées (bibliographie, remerciements) ; vide = tout indexer |
| `HGRAG_TABLE_MODE` | `accurate` | Modèle de structure des tableaux Docling (`accurate` ou `fast`) |
| `HGRAG_TABLE_CELL_MATCHING` | `1` | Associe les cellules détectées au texte du PDF |
| `HGRAG_FORMULA_MODE` | `vlm` | Formules → LaTeX : `vlm` = modèle de vision par l'API après le parsing (quelques secondes par formule, `formulas.json`), `docling` = CodeFormulaV2 exécuté par Docling sur CPU (≈ 1 min par formule), `off` |
| `HGRAG_FORMULA_VLM` | = `HGRAG_FIGURE_VLM` | Modèle utilisé pour les formules en mode `vlm` |
| `HGRAG_FORMULA_READING` | `1` | Après le LaTeX, une lecture de chaque formule en anglais courant (`HGRAG_TABLE_MODEL`, texte seul : le LaTeX et les phrases autour), ajoutée au texte vectorisé du chunk de formule (`formulas.json`, `reading`). Une question qui paraphrase une formule (« quand le rapport est négatif ») n'a aucun mot commun avec son LaTeX |
| `HGRAG_CODE_ENRICHMENT` | `0` | Blocs de code : langage et indentation. Désactivé : Docling exécute CodeFormulaV2 sur CPU (≈ 170 s par bloc) et boucle souvent sur des blocs mal détectés (`TailRepetitionStopper`) |
| `HGRAG_PICTURE_CLASSIFICATION` | `0` | Type de figure (graphique, schéma…) dans les métadonnées |
| `HGRAG_TABLE_IMAGES` | `0` | Exporte une image PNG de chaque tableau |
| `HGRAG_LLM_SEND_IMAGES` / `HGRAG_LLM_MAX_IMAGES` | `1` (Claude) / `6` | Figures envoyées comme images au LLM de réponse |
| `HF_TOKEN` | — | Jeton HF pour `hgrag vlm-bench` (router d'inférence) |
| `HGRAG_JUDGE_MODEL` | `claude-haiku-5-5` | Modèle du juge de `hgrag eval` |
| `HGRAG_REFINE_MODEL` | `claude-haiku-5-5` | Modèle des juges de contexte et de génération (`hgrag ragbench-refine`) |
| `HGRAG_REFINE_THRESHOLD` | `4` | Questions notées strictement en dessous jugées à nouveau |
| `HGRAG_REFINE_CONTEXT_OK` | `4` | `context_score` minimal pour lancer le juge de génération |
| `HGRAG_EVAL_DIR` | `data/eval` | Résultats et store du benchmark |
| `HF_HUB_OFFLINE` | — | `1` = ne plus contacter Hugging Face (modèles déjà en cache) |
| `HGRAG_RUNS_DIR` | `data/runs` | Statuts et logs des runs de l'app |
| `HGRAG_PYTHON` | `<racine>/.venv/bin/python` | (Airflow, plus tard) interpréteur du pipeline |
| `ANTHROPIC_API_KEY` | — | Clé Claude API (génération de la réponse) |
| `HGRAG_LLM_PROVIDER` | `anthropic` si la clé existe, sinon `openai` | `anthropic` ou `openai` (serveur local compatible) |
| `HGRAG_LLM_MODEL` | `claude-haiku-5-5` / `llama3.1` | Modèle de génération |
| `HGRAG_LLM_BASE_URL` | `https://api.anthropic.com` / `http://localhost:11434/v1` | Point d'accès |
| `HGRAG_LLM_MAX_TOKENS` | `2000` | Longueur maximale de la réponse (réessai à ×2 dans ragbench si la réponse est vide ou coupée) |
| `HGRAG_LLM_MAX_CONTEXT_CHARS` | `48000` | Budget de caractères des sources dans le prompt (à 24000, la section de référence était coupée pour 6 questions du ragbench ; à 48000, pour aucune) |
| `HGRAG_RAGBENCH_DISPUTED` | `data/ragbench_disputed.json` | Questions du ragbench à référence fausse (vérifiée dans le PDF), exclues du bloc `excluding_disputed` |
| `MLFLOW_TRACKING_URI` | SQLite `data/mlflow/mlflow.db` | Backend MLflow |

Les options Docling (`HGRAG_TABLE_*`, `HGRAG_FORMULA_MODE=docling`, `HGRAG_CODE_ENRICHMENT`, `HGRAG_PICTURE_CLASSIFICATION`) sont enregistrées dans `manifest.json` (`parse_options`) ; `ragbench` ne réutilise un parsing que s'il a été fait avec les mêmes options.

#### Premier téléchargement des modèles (formules, figures)

Avec `HF_HUB_OFFLINE=1`, Docling ne peut pas télécharger un modèle absent du cache. Avec `HGRAG_CODE_ENRICHMENT=1` ou `HGRAG_FORMULA_MODE=docling`, Docling charge
`docling-project/CodeFormulaV2` (et `HGRAG_PICTURE_CLASSIFICATION` un classifieur de figures). Une seule fois, en ligne :

```powershell
$env:HF_HUB_OFFLINE = "0"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/CodeFormulaV2')"
```

Ensuite on peut remettre `HF_HUB_OFFLINE=1`.

Même chose pour le modèle d'embedding et le reranker (par défaut) :

```powershell
$env:HF_HUB_OFFLINE = "0"
uv run python -c "from huggingface_hub import snapshot_download; [snapshot_download(m) for m in ('BAAI/bge-base-en-v1.5','cross-encoder/ms-marco-MiniLM-L12-v2')]"
```

### Notes
- La couche Neo4j est volontairement laissée de côté pour l'instant.

---

## 5. Brancher Airflow plus tard

Les étapes (`hgrag.pipeline` : `parse_step`, `build_and_save_hierarchy`, `index_documents`, et
`hgrag.retrieval`) sont les mêmes dans l'exécuteur local et dans les DAGs de `orchestration/`
(`hgrag_pipeline`, `hgrag_retrieval`, graphe identique, suivi MLflow identique via le `run_id`).
Quand les workflows seront stabilisés : `cd orchestration && uv sync && ./run_airflow.sh` (WSL,
port configurable via `AIRFLOW__API__PORT` si 8080 est pris, LocalExecutor limité à 4 workers par
défaut). `app/airflow_client.py` est un client REST Airflow 3 prêt à servir pour que la page
*Pipelines* affiche les runs Airflow au lieu des runs locaux.
