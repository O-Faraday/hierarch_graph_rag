# Hierarchical RAG: a tour of the Streamlit app

Part 2 of a series on multimodal PDF pipelines. 
[Part 1](https://medium.com/@oliver-mlops/the-first-step-of-a-multimodal-pipeline-which-pdf-parser-keeps-the-structure-303a9968e39f) asked which parser keeps the structure. This one is about what I built on top of it.

In the first article I compared PDF parsers on one question: which one keeps the structure of a document?

This article is a tour of the Streamlit app I use to run the pipeline, look inside every step, and ask questions. 
I built it because a RAG pipeline that you cannot see is a RAG pipeline you cannot fix. 
Every feature below exists because I got stuck without it.

The code is on GitHub: (https://github.com/O-Faraday/hierarch_graph_rag).

## 1. The idea in one paragraph

Most RAG tutorials cut a document into content chunks and forget where each chunk came from. 
Here I use Docling to parse the PDF with its hierarchical option. Each chunk remembers its place in that building tree. 
At query time I find the best chunks, and  **rebuild the whole section**: paragraphs, tables and figures in their original order. 

The first page of the app is the data pipeline steps :

- **Parse**: Docling extracts text, tables and figures. Figures are saved as PNG files with their metadata and description
- **Hierarchy**: the headings become a tree, and every element (paragraph, table, figure, formula) hangs from its section.
- **Index**: paragraphs are split into chunks, embedded with a sentence-transformer model, and stored in a Qdrant collection together with their section path.

The app has 2 more pages **Query** and **VLM comparison**.

## 2. Pipelines: launching a run

The first page is where the vector index is set.

**Pick a folder.** A text field points to the folder of PDFs. A button next to it switches to three *control PDFs*. They are my smoke test.

**Describe figures and tables (optional).** A checkbox asks a vision-language model through the Hugging Face API, to describe each figure, describe each table, and transcribe each formula as LaTeX.

**Three buttons to refresh the pipeline steps.**

- *Run whole pipeline*: parse, hierarchy, index.
- *Rebuild hierarchy + index*: reuse the parsing on disk.
- *Re-index only*: re-chunk and re-embed, for example after changing the embedding model or the chunk size.

**A custom run.** An expander lets me choose the documents and the steps one by one, and cancel a run in progress. Each run is a separate process, so the app stays responsive while it works.

*Figure 1: the launch area. Folder, control PDFs, description checkbox, and the three run buttons.*

## 3. Pipelines: watching a run

This is the part I am proudest of.

The page shows a **run board**: documents as rows, steps as columns, one cell per document and step. A cell tells you if it is waiting, running, done, reused from disk or failed, with its duration and, for a failure, the error message. A thin progress rail sits under each column header. The index step is a single merged cell, because it works on the whole collection at once.

Before the board:

- a **header** with the state of the run, its start time, elapsed time, the steps it ran and its scope;

Under the board:
- a **tasks table**. I can tick rows in it and press *Re-run selected rows*, which restarts the chosen documents from the earliest selected step;


*Figure 2: the run board during an indexing run. One row per paper, one column per step.*


## 4. Pipelines: looking inside

A run that succeeds can still produce garbage, so the page lets me open the products of each step.

**Documents on disk** lists every PDF with its parsed and hierarchy status.

**Inspect a document** has four tabs:

- *Parse*: the manifest (page count, options used, number of figures and tables), a bar chart of how many items Docling found per label, and the Markdown export.
- *Hierarchy*: the heading tree as plain text. Most of my bugs were visible here at a glance, for instance a section title detected one level too deep.
- *Chunks*: every chunk in a table, with its section path.
- *Figures*: each extracted PNG next to its metadata and description.

**Control check** runs the three control PDFs through the pipeline and compares the result with what I expect: headings and levels, figures, tables, chunks, and which section comes back for each needle query.

**Export for analysis** builds one zip with the Docling outputs, the last run logs, the MLflow database, the library versions and the structural check. 

*Figure 3: the Hierarchy tab. The heading tree of one paper, with the figure and table leaves.*

## 5. Query: the retrieval, step by step

The second page answers questions.
First, an expander picks the model: the Claude API, or any OpenAI-compatible server such as Ollama or LM Studio. API keys can be typed for the session only; they are never written to disk or to MLflow.
I type a query, choose *k* chunks to retrieve, and optionally include sub-sections. Then the page streams each step as it completes:

1. **Embed the query.** The model name, the vector dimension, the time in milliseconds, and a small chart of the first components of the vector.
2. **Vector search.** The best chunks with a score bar, their section path, their page and their text.
3. **Group by section.** The chunks are merged by section, so one section with three matching chunks ranks as one result.
4. **Reconstruct.** Each section is rebuilt in full, in document order, with its figures displayed under the text.
5. **Answer.** A language model reads the rebuilt sections and answers, streamed token by token. Below the answer I show the numbered sources with their section and score, the number of tokens, the number of figure images sent, and the time to the first token.


*Figure 4: the LLM and the query.*

## 6. VLM comparison: which model reads a figure best?

The third page started as a side project. When figures became searchable chunks, the quality of their descriptions became a retrieval problem, so I needed to measure it.

The page runs several vision-language models on the control figures, through the Hugging Face router and the Claude API, so nothing is installed locally. For each figure I wrote a list of known facts: the chart type, the title, the axes, a few values and the trend. A Claude judge checks every description against these facts and flags what the model **invented**.

The results come as:

- a **leaderboard** with a score from 1 to 5, the share of facts found, the number of invented claims per figure, the time and the output tokens;
- two **bar charts**, facts found and inventions per model;
- a **figure-by-figure** view, with the image on the left and every model's description on the right, with ticks for the facts it found.

You can add any model by name, list the models served by the router, edit the prompt, and turn the judge off. This page is how I chose the model that now describes my figures.

*Figure 5: the leaderboard. Score, facts found and inventions per model.*

## 7. Limits, stated plainly

- The app is a development tool for one user. It has no authentication and no multi-user mode.
- Qdrant runs in local mode, so there is one process at a time.
- The embedding model is small and English-centric. A better one is the obvious next upgrade.
- Tables split across pages are still chunked as two tables.
- The orchestration is a simple local runner. An Airflow version exists in the repository, but I have not put it behind the app yet.

## 8. What comes next

The next article will cover the evaluation against an evaluation dataset.
If you have run a pipeline on this benchmark and have numbers to compare, I would like to hear from you.
