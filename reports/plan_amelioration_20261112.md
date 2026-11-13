# Plan d'amélioration : mesure, recherche, qualité, cycle de vie Databricks

12 novembre 2026. Revue du projet après les runs QASPER sur 100 articles
(`reports/qasper_dev100_20261110.md`). Les gains annoncés ci-dessous sont des hypothèses à mesurer, pas des résultats.

## 1. État du projet

- **Solide :** parsing et arbre (signets, réparations Docling), reconstruction des sections, traces MLflow, contrôles
  sans référence (`hgrag assess`), 426 tests hors ligne.
- **Recherche :** avec 10 sections et 50 candidats, la section de la preuve est lue dans 90 % des cas sur QASPER.
  Deux faiblesses restent : les figures et tableaux (hit@10 0,65 contre 0,76 pour le texte, avant le passage à 10
  sections), et les sections de synthèse (abstract, introduction, conclusion) qui sortent en tête pour 37 % des questions.
- **Génération :** c'est elle qui limite le score. À contexte presque identique, une quarantaine de questions montent
  et autant descendent d'un run à l'autre. Tant que ce bruit n'est pas réduit, un gain de moins de 3 à 5 points n'est
  pas mesurable.
- **Databricks :** le code existe (bundle, 4 notebooks, jobs d'ingestion et d'arrêt de l'index, `hgrag register`,
  `hgrag evaluate-rag`), mais rien n'a encore tourné sur un workspace. Tout y est testé avec des faux.

## 2. GNN et index des chunks

**GNN : pas maintenant.**
- Un GNN s'entraîne sur des paires question / section pertinente. Nous n'en avons que quelques centaines (QASPER),
  dans un seul domaine.
- Le graphe est petit : un arbre par document.
- Un GNN ajoute un modèle à entraîner, versionner et servir.
- Les travaux comparables (SF-RAG, HippoRAG2) obtiennent leurs résultats sans GNN, par navigation dans l'arbre ou
  par PageRank personnalisé (chiffres annoncés par les auteurs, non reproduits ici).

**À la place, du moins cher au plus cher :**
1. **Arêtes de citation :** relier un tableau ou une figure à la section qui le cite (« Table 3 shows… ») et
   l'ajouter au contexte quand cette section est retenue. Cela vise directement la faiblesse figures / tableaux.
2. **Index à deux niveaux :** en plus des chunks, indexer un résumé par section, chercher d'abord les sections, puis
   les chunks dedans (la navigation « de haut en bas » de SF-RAG).
3. **Diffusion de score dans l'arbre :** un chunk bien noté remonte un peu ses sections sœurs et parente. C'est
   l'idée utile d'un GNN, sans entraînement, avec un seul paramètre à régler.
4. **GNN**, seulement si 1 à 3 plafonnent et qu'on dispose de milliers de paires étiquetées (par exemple les retours
   utilisateurs collectés sur Databricks).

## 3. Tâches

« Modèle » : le modèle qui réalise la tâche de développement. Quand la tâche ajoute un appel LLM au pipeline, le
modèle d'exécution est indiqué aussi.

### A. Fiabiliser la mesure

| # | Tâche | Mesure du succès | Modèle |
|---|---|---|---|
| A1 | Écart apparié et erreur type entre deux runs, intégrés à `ragbench-report` (aujourd'hui un script à part) | Chaque comparaison affiche « +x ± y » | Sonnet 5.5 |
| A2 | Mesurer le bruit de génération : 3 runs identiques sur les 100 articles, contexte figé | Écart-type du F1 entre runs connu | Sonnet 5.5 ; exécution Haiku 5.5 |
| A3 | Liste des références QASPER contestées et métriques « hors contestées » | Bloc `excluding_disputed` pour QASPER | Opus 5.5 (lecture critique des références) |
| A4 | Vérifier `HGRAG_K_SECTIONS=10` sur `data/pdf_eval`, puis en faire le défaut | Pas de recul sur pdf_eval | Sonnet 5.5 |

### B. Recherche et index

| # | Tâche | Mesure du succès | Modèle |
|---|---|---|---|
| B1 | Arêtes de citation tableau / figure → section | Hit@10 et note du juge sur les 48 questions figures / tableaux | Opus 5.5 |
| B2 | Index à deux niveaux avec résumés de section | Hit@1, MRR, part des sections de synthèse au rang 1 | Opus 5.5 ; résumés par Haiku 5.5 en batch |
| B3 | Diffusion de score dans l'arbre | Hit@10 à nombre de sections égal | Sonnet 5.5 |
| B4 | Descriptions de courbes qui relèvent les valeurs lues | Note du juge sur les questions à valeurs lues sur figure | Sonnet 5.5 ; exécution : modèle de vision actuel, ou Sonnet 5.5 en comparaison |

### C. Qualité des réponses

| # | Tâche | Mesure du succès | Modèle |
|---|---|---|---|
| C1 | Sortie structurée pour QASPER (type de réponse + réponse) au lieu d'un format imposé par le prompt | F1 des « X ou Y ? » et des listes | Sonnet 5.5 ; exécution Haiku 5.5 |
| C2 | Comparer Haiku 5.5, Sonnet 5.5 et Opus 5.5 en génération, contexte figé, après A2 | Écart apparié supérieur au bruit, ou conclusion « pas de gain » | Sonnet 5.5 pour le harnais |
| C3 | Vérification de la réponse contre les sources avant de la rendre (contrôle `answer` existant, rendu bloquant en option) | Taux de réponses non étayées | Opus 5.5 ; exécution Haiku 5.5 |

### D. Cycle de vie Databricks

| # | Tâche | Mesure du succès | Modèle |
|---|---|---|---|
| D1 | Premier déploiement réel en `dev` : parsing, index AI Search, question, sur les 9 PDF de `pdf_eval` | Mêmes réponses qu'en local sur les 61 questions | Opus 5.5, avec vous (catalogue, schéma, volume, endpoint à fournir) |
| D2 | Modèles enregistrés dans Unity Catalog avec alias `champion` / `challenger`, promotion conditionnée à `evaluate-rag` | Une promotion refusée quand le score baisse | Opus 5.5 |
| D3 | Job d'évaluation planifié sur le jeu de référence, résultats dans une table Delta et un tableau de bord | Historique des métriques par version | Sonnet 5.5 |
| D4 | Suivi en production : contrôles sans référence échantillonnés sur les traces, alerte sur dérive | Alerte déclenchée sur un cas injecté | Sonnet 5.5 ; juges Haiku 5.5 |
| D5 | Collecte des retours utilisateurs (pouce, correction) rattachés à la trace, versés au jeu d'évaluation | Retours visibles dans MLflow et réutilisables | Sonnet 5.5 |
| D6 | CI : tests et `bundle validate` à chaque push, déploiement `prod` sur tag | Pipeline vert, déploiement reproductible | Sonnet 5.5 |

## 4. Choix des modèles

Prix par million de tokens (entrée / sortie), table de référence du 3 novembre 2026 :

| Modèle | Prix | Usage ici |
|---|---|---|
| Haiku 5.5 (`claude-haiku-5-5`) | 0,10 $ / 0,50 $ | Exécution en volume : génération, juges, résumés |
| Sonnet 5.5 (`claude-sonnet-5-5`) | 2 $ / 10 $ | Développement bien spécifié |
| Opus 5.5 (`claude-opus-5-5`) | 4 $ / 20 $ | Conception et tâches à jugement (B1, B2, D1, D2, A3) |
| Fable 5.2 (`claude-fable-5-2`) | 8 $ / 40 $ | Aucune tâche de cette liste ne le justifie |

Fable 5.2 ne se justifierait que pour une refonte d'architecture menée en autonomie sur plusieurs heures.

## 5. Résultats (13 novembre 2026)

**A1.** `hgrag ragbench-compare <run_A> <run_B>` : écart apparié de chaque mesure, erreur type par article, verdict.

**A4.** `hgrag.toml` passe à 10 sections et 50 candidats. `data/pdf_eval` : recherche identique (hit@1 0,793,
hit@10 0,914), juge 4,90 → 4,89.

**A2.** Trois runs identiques (100 articles QASPER, 10 sections, prompt v2, Haiku), contexte identique (0 question
changée d'un run à l'autre) :

| Run | F1 | Juge |
|---|---|---|
| `20261111T015032-c1543e` | 0,553 | 4,10 |
| `20261113T110636-e28ea8` | 0,536 | 4,07 |
| `20261113T084512-8d6ca0` | 0,535 | 4,08 |
| **Écart-type entre runs** | **0,010** | **0,017** |

- 132 questions sur 334 changent de F1 de plus de 0,05 entre les trois runs, et 54 de plus de 0,3.
- Les plus instables sont les questions sans réponse (F1 0,54 / 0,46 / 0,39 sur 28 questions) et les oui/non.
- Conséquence : le run b était un tirage haut. Le gain du nouveau prompt mesuré sur un seul run (+0,014) est de
  l'ordre du bruit ; seul le gain de recherche est établi.
- Pour juger un changement de génération, comparer la moyenne de 2 ou 3 runs, ou viser un écart supérieur à 0,03.

## 6. Ordre conseillé

1. **A1, A2, A4** : sans mesure du bruit, on ne peut pas juger les tâches B et C.
2. **D1** : c'est le plus gros risque non levé du projet.
3. **B1**, puis **B2** : les deux gains de recherche les plus probables, aucun n'est mesuré à ce jour.
4. **D2 à D6**, en parallèle de **C1 à C3**.
