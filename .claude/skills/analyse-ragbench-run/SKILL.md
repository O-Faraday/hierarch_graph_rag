---
name: analyse-ragbench-run
description: Analyse un run du ragbench (summary.json, questions.md, results.jsonl) pour séparer les échecs de recherche des échecs de génération et proposer des correctifs. À utiliser quand l'utilisateur fournit les fichiers d'un run.
argument-hint: <dossier du run ou fichiers>
---

# Analyser un run ragbench

1. **Résumé** : lire `summary.json`. Rapporter doc hit@1/@10, section hit@1/@10, MRR, context recall, note moyenne et part ≥ 4, par source (`text`, `text-image`, `text-table`, `text-table-image`) et par type (`extractive`, `abstractive`). Comparer avec le run de référence précédent (même échantillon : `config.key`).
2. **Rangs** : dans `results.jsonl`, compter `section_rank` (1, 2, 3…, absent) et `doc_rank`. Beaucoup de rangs 2 à 4 = problème de classement (reranker, embedder) ; section absente = problème de recherche ou de découpage.
3. **Recherche contre génération** : pour les notes ≤ 2, séparer les cas où la bonne section est dans le contexte (`section_rank` non nul : échec de lecture ou de génération) de ceux où elle est absente.
4. **Cas types** : questions sur figures sans valeurs écrites ; questions génériques que le papier ne traite pas (réponse attendue hors document) ; sens inversé ; réponse de référence douteuse (vérifier avec `gold_text` avant d'accuser le pipeline).
5. **Trois exemples à regarder en détail** : id, papier (`data/ragbench/pdfs/<id arXiv>.pdf`), question, attendu, réponse, commentaire du juge, section retrouvée.
6. **Correctifs** : classés par gain probable et par coût, sans annoncer de gain non mesuré.

Rappels : les métriques sont maison (pas celles de Vectara) ; le juge peut se tromper, relire `rationale` ; ne jamais exposer de clé.
