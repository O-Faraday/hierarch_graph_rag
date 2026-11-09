# Where the answers get lost: evaluating a hierarchical RAG on Open RAG Bench

Part 3 of a series on multimodal PDF pipelines.
[Part 1](https://medium.com/@oliver-mlops/the-first-step-of-a-multimodal-pipeline-which-pdf-parser-keeps-the-structure-303a9968e39f) asked which parser keeps the structure. Part 2 toured the Streamlit app I built on top of it. This one puts numbers on the whole chain, and follows the failures down to their cause.

The code is on GitHub: (https://github.com/O-Faraday/hierarch_graph_rag).

At the end of Part 2 I promised an evaluation. I expected to tune a prompt or two.
Instead, the evaluation taught me three things I did not expect:

- The judge was wrong more often than I thought.
- The parser was almost never the problem.
- Answers were lost at the cut between the search and the reranker more often than in any model.

And once the benchmark had told me what it could, I needed checks that work without a reference answer, for the documents that will never have one.

## 1. The benchmark and what I measure

I use [Open RAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) from Vectara. It is a set of arXiv papers with text, tables, figures and formulas. Each question comes with a reference answer and the section of the paper that holds the evidence. Questions are tagged by source: *text*, *text-image*, *text-table* and *text-table-image*.

My sample has **60 papers**: 30 papers with questions and 30 *negatives*, papers with no question that act as distractors. That makes **233 questions** and about 11,000 chunks in one Qdrant collection.

The benchmark parses the papers with its own parser, not mine, so I cannot match sections by index. I match them by content. All the metrics below are mine, not the official Vectara ones:

- **Doc hit**: the right paper is among the retrieved sections.
- **Section hit@1 and @10**: a retrieved section is "on the reference" if one of its chunks shares at least half of its word 5-grams with the reference text, or if its heading is the reference heading. The heading rule exists because LaTeX written by two parsers breaks the word overlap.
- **MRR**: the mean of 1/rank of the reference section.
- **Context recall**: the share of the reference text found in the sections I rebuild for the language model. This is where a hierarchical RAG should shine.
- **Judge score**: Claude Haiku compares the generated answer with the reference answer and gives a score from 1 to 5, with a two-line rationale.

Open RAG Bench is under CC BY-NC 4.0. I use it as an internal validation tool, nothing more.

## 2. The first numbers

The baseline chain is the one from Part 2: Docling parsing, a heading tree, paragraph chunks embedded with `all-MiniLM-L6-v2`, the 30 closest chunks reranked by a cross-encoder (`ms-marco-MiniLM-L12-v2`), the best 10 kept, their sections rebuilt in full, and Claude Haiku answering from them.

| Metric | Value |
|---|---|
| Doc hit@1 | 98.7 % |
| Section hit@1 | 71.7 % |
| Section hit@10 | 90.6 % |
| MRR | 0.795 |
| Context recall | 0.715 |
| Mean judge score | 4.15 / 5 |
| Answers scored 4 or 5 | 84.5 % |

The right paper is almost always found. The right section is first seven times out of ten. Figures are the weak spot: section hit@1 is 80 % on *text* questions, 58 % on *text-image* and 48 % on *text-table-image*.

So I started with the figures.

*Figure 1: the Results page. Every question of a run in one table, lowest score first.*

## 3. Reverse engineering the figure failures

I took the eleven worst questions that my rule-based diagnosis labelled "figure reading". For each one I knew the paper, the question and the expected answer. I checked whether the parser had brought up the context in which the answer can be found.

**Four of the eleven were not about figures at all.** My rule matched the word "graph" in "de Bruijn graph" and "Figure 7" in a citation. My own diagnostic tool was the first thing to fix.

**On the seven real figure questions, Docling did its job.** The figures were cropped in full, attached to the right section, and sent as images to the language model. The failures came from elsewhere:

| What went wrong | Questions |
|---|---|
| The measure: an imprecise reference, or a judge too literal | 3 |
| The generation: the model added its own reading of a chart against a value written in the text, or answered "No" first | 2 |
| A grey-scale stacked bar chart that nobody could read, me included | 1 |
| The vision model's description: it took the colour scale for an axis | 1 |

One example sums it up. The reference answer says −80 dBm. The chart shows −77 dBm. The model answered −77 dBm, and the judge gave it a 2.

The judge only sees the question, the reference answer and the generated answer. It cannot tell a wrong answer from a wrong reference, or a retrieval failure from a generation failure. I needed a second opinion.

## 4. A second opinion, in two steps

For every question scored below 4, two more judges run. Both see the **whole PDF**. They started on Claude Sonnet; they now run on Claude Haiku, like the generator and the first judge, so the whole chain uses one model.

1. **The context judge** reads the PDF and the exact context the generator received: the sections and the figures. It scores how far that context holds what is needed to answer, from 1 to 5, and says where the PDF answers.
2. **If the context scores 4 or more**, the **generation judge** grades the answer again, with the PDF and the context at hand. It follows the PDF when the reference is imprecise, and accepts a value read from a chart within about 5 %. It gives a refined score and a cause: *reference*, *judge*, *generation* or *none*.
3. **If the context scores below 4**, the first score is kept and the cause is *context*: a retrieval or parsing failure.

The first score is never overwritten. Both are reported. A second LLM judge can be more lenient than the first, and I want to see the gap, not hide it.

The PDF, the figures and the context go first in the request, with a prompt-cache breakpoint, so the second call reads them from the cache.

On one run, with the same retrieval but answers written by Claude Sonnet that time, 37 questions scored below 4:

| Cause | Questions |
|---|---|
| Context insufficient (retrieval or parsing) | 14 |
| First judge too strict | 11 |
| Reference answer wrong or imprecise | 10 |
| First score stands | 2 |

With the refined scores, answers scored 4 or 5 go from 84 % to 93 %. Read that number with care: it is the opinion of a second model, not a fact. What it does give me is a list. Fourteen questions where the evidence never reached the model. Those are the ones worth fixing.

*Figure 2: the second opinion in the Question detail page. Context score, refined score, cause, and both rationales.*

## 5. Fourteen context failures, three under the microscope

I took three questions whose refined score stayed at 2, all with the cause *context*:

- **A.** How do normal, Laplace and Cauchy distributions affect the performance of step size selectors in MCMC?
- **B.** How do different enhancement strategies affect the output quality of generated packages?
- **C.** What role do positive or negative ratios play in the behaviour of an envelope function?

Then I replayed the retrieval with the real models, stage by stage.

**The parsing was right in all three.** Section 5.1 of paper A, with its two paragraphs on Laplace and Cauchy targets and its Figure 1, was there and cleanly chunked. Section 4.7 of paper C was there with its formulas.

**The cut was the problem.** The vector search sends its 30 best chunks to the reranker. In paper A, the chunks of the right section came in at ranks 32, 37, 38 and 40. In paper B, the paragraph holding the answer came in at ranks 34 to 39. Just outside.

When I forced those chunks into the reranker, it put them **first**: 0.81 for A, 0.99 for B. The reranker knew. It never saw them.

Why did the embedding miss them? The question words "step size selector" appear in thirty chunks of paper A. The words that matter, *Laplace* and *Cauchy*, are rare, and a small dense embedding dilutes them.

Paper B had a second problem, and this one was Docling's. The title printed inside a bar chart, *"Enhanced Feature and Description Review Scores by llama-3.1-70b-versatile"*, was read as a section heading. Section 6.3 was cut in two: one introductory paragraph on one side, the ablation table and its discussion under the fake heading on the other.

*Figure 3: rank of the reference section at each stage. Dense search, keyword search, fusion, reranker.*

## 6. Hybrid search: 30 % keywords

If the problem is rare words, keywords are the cheap answer.

Each chunk now carries a second vector in Qdrant: a sparse **BM25** vector. I compute the term weights, saturated and normalised by the chunk length. Qdrant multiplies them by each term's inverse document frequency over the whole collection, with its IDF modifier. English stop words are removed, and terms are hashed, so there is no vocabulary to store.

At query time I run both searches and fuse them: each list's scores are normalised between 0 and 1, then combined as **70 % dense and 30 % keywords**. The fused candidates go to the reranker as before.

On a copy of the MiniLM index, with the keyword vectors added and no re-embedding (75 seconds for 11,000 chunks):

| Question | Rank of the right chunk, dense | Keywords | After the reranker, 30 candidates |
|---|---|---|---|
| A (Laplace, Cauchy) | 32 | **2** | section found, **first** |
| B (enhancement) | beyond 30 | **1** | section found, **first** |
| C (envelope) | 14 | 16 | not found |

Two out of three fixed, with the same 30 candidates and the same models.

Question C did not move. It shares no word with its answer.

## 7. The formula nobody could find

Section 4.7 of paper C answers question C in one formula:

```
E(t) = C / Γ(t/τ + 1) · (b/(aτ))^(t/τ)        if b/τ > 0
E(t) = ± C / Γ(t/τ + 1) · (−b/(aτ))^(t/τ)     if b/τ < 0
```

The question says the same thing in words: "positive or negative **ratios**". The benchmark's questions are written by a model that read the formula and paraphrased it.

Neither search can bridge that. For the keyword search, the LaTeX becomes the tokens `b`, `tau`, `0`, `frac` and `gamma`, with no word in common with the question. For the embedding, LaTeX is close to noise: the formula chunk had a cosine of 0.37 with the question, under the 0.40 needed to reach the top 30. The sentence that explains the formula sits in another chunk and says nothing about signs.

So I tested the idea by hand. I wrote the formula as two sentences and embedded them. They scored **0.63**, above the best chunk of the whole index (0.52). I wrote those sentences knowing the answer, so this was a direction, not a measure.

The real fix is a **reading** of every formula. After the vision model transcribes a formula into LaTeX, a text model gets the LaTeX and the sentences around it, and writes at most two plain-English sentences. The reading is added to the text that is embedded and indexed. The chunk itself stays LaTeX, for display and for the language model. This is what it wrote for the formula above, without seeing the question:

> The formula gives the value of E(t), which approximates the envelope of the waveform X(t) when the parameter τ is large. If the ratio b/τ is positive, E(t) equals c divided by the gamma function of (t/τ + 1), multiplied by (b/(aτ)) raised to the power of t/τ; if b/τ is negative, E(t) equals plus or minus the same expression…

Re-indexed, the formula chunks entered the candidates: 4th and 5th in the keyword search. The answer still did not come back.

**The reranker was blind.** It reads the headings and the chunk text, which is the LaTeX, not the reading. It gave the formula chunks 0.000. It did the same to tables in Markdown: the ablation table of paper B had scored 0.000 as well.

Once the reranker also read the reading, between the headings and the text, the formula chunk went from **0.000 to 0.43**. The best competing chunk was at 0.055.

The lesson cost me a run: **what you embed, what you match by keyword and what you rerank must be the same text.** I had enriched the first two and forgotten the third.

## 8. Results

On the three questions, through the whole chain (retrieval, reranker, Haiku, judge):

| Question | Before | After |
|---|---|---|
| A (Laplace, Cauchy) | section not found, score 2 | section first, score **4** |
| B (enhancement) | section not found, score 2 | section first, score **4**, with the ablation table in the answer |
| C (envelope) | section not found, score 2 | section first, score **4**, both cases and the limits at t = 0 |

Three questions chosen because they failed do not make a benchmark. Keywords can also pull a distractor paper up. So I re-ran the whole sample from the existing parsing: every figure, table and formula described again, the hybrid index rebuilt, all 233 questions answered and judged, and the second opinion on the low scores.

Same embedding model (MiniLM), same reranker, same 30 candidates, same generator (Haiku):

| Metric | Baseline | New chain | Difference |
|---|---|---|---|
| Section hit@1 | 71.7 % | 73.4 % | +1.7 pts |
| Section hit@10 | 90.6 % | 93.6 % | **+3.0 pts** |
| MRR | 0.795 | 0.817 | +0.022 |
| Context recall | 0.715 | 0.740 | +0.025 |
| Mean judge score | 4.15 | 4.17 | +0.02 |
| Answers scored 4 or 5 | 84.5 % | 85.3 % | +0.8 pt |

The retrieval moved: **8 sections found that were missed before, 1 lost**, and that one still got a 5. These numbers depend on the retrieval only, so they are repeatable.

The judge scores barely moved, and I do not trust them to. On the 80 questions whose context was **exactly the same** in both runs, 28 scores still changed, 12 up and 16 down. That is the noise of one generation and one judgement. A difference of 0.02 is inside it, in either direction.

What I looked for were regressions. Ten answers went from 4 to 3. In nine of them the section came back at the same rank as before; in the tenth it slipped from 3rd to 4th. The second opinion raised all ten back to 4 or 5 and named the same cause each time: the first judge. Among the 34 answers below 4, it found 7 where the evidence never reached the model, against 14 on the earlier run. With its scores, 96 % of the answers reach 4 or more, against 85 % for the first judge. Same caution as before: that is a second model's opinion.

So the honest summary is this. The new chain finds more of the right sections and breaks nothing I can see. Whether that turns into better answers on this sample is below what one LLM judge can measure.

*Figure 4: the Compare runs page. Settings that changed, metrics side by side, then every question that moved, regressions first.*

## 9. The app grew a comparison page

A run that improves the mean can still break twenty questions. So the app now has a **Compare runs** page. It takes a reference run and a new run of the same bench. Then it shows, on their common questions:

- the settings that differ: embedder, keyword weight, reranker, models;
- the metrics side by side, with the difference, for the first judge and the refined scores;
- counts of questions fixed (now 4 or more), regressed (now below 4), better and worse, and of sections gained and lost;
- every question that moved, regressions first. One click opens it in the detail page.

The detail page also shows the second opinion: the context score, the refined score, the cause and both rationales.

## 10. The two settings I had not tuned

The keyword weight (30 %) and the number of candidates (30) were first guesses. I swept them on all 233 questions. These metrics depend on the retrieval only, so no language model is involved and the numbers repeat.

| Setting | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| **30 candidates, 30 % keywords** | **73.4 %** | **90.1 %** | **93.6 %** | **0.817** |
| No keywords | 72.5 % | 88.4 % | 90.6 % | 0.800 |
| 50 % keywords | 73.4 % | 90.6 % | 93.6 % | 0.817 |
| 50 candidates | 73.0 % | 89.7 % | 94.0 % | 0.814 |
| 100 candidates | 73.0 % | 89.7 % | 94.0 % | 0.815 |
| Summary sections × 0.9 | 69.1 % | 85.4 % | 93.6 % | 0.782 |

The keywords earn their place: without them, hit@10 loses 3 points. Beyond that, nothing moves. More candidates give the reranker more chances to be wrong as well as right.

The last line tests an idea of mine. Abstracts and conclusions win many queries because they summarise everything, so I lowered their scores by 10 %. It cost 4 points of hit@1. They win because they often hold the answer.

I kept 30 and 30 %.

## 11. The last answers below 4

On the full run, 34 answers scored below 4 with the first judge, and I read each one. They fall into four groups.

**The prompt budget.** The rebuilt sections go to the model inside a budget of 24,000 characters. For six questions, the reference section was found, then cut from the prompt because the sections ranked above it filled the budget. The ten sections weigh 23,000 characters at the median and 44,000 at the 90th percentile, so I raised the budget to 48,000. No reference section is cut any more. On the fourteen questions I re-ran, two moved from 2 and 3 to 4. Fourteen questions are too few to call that a gain; the cut itself is gone, and that I can check.

**The reranker.** For twenty-three questions, the reference section was ranked below third, or not found at all. I tried two stronger rerankers on them.

| Reranker, on the 23 questions outside the top 3 | Hit@3 | Hit@10 | Time per question |
|---|---|---|---|
| ms-marco-MiniLM-L12-v2 (current) | 0 % | 34.8 % | a few seconds |
| bge-reranker-v2-m3 (local CPU) | 17.4 % | 43.5 % | about 26 s |
| Qwen3-Reranker-0.6B (Hugging Face API) | 30.4 % | 39.1 % | about 10 s |

On the hard questions, Qwen3 looked like the answer. So I ran it on all 233:

| | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| MiniLM-L12 | 73.4 % | **90.1 %** | 93.6 % | **0.817** |
| Qwen3-Reranker | **74.2 %** | 88.4 % | 93.6 % | 0.814 |

31 sections moved up and 32 moved down. Qwen3 rescued questions that MiniLM had buried, and pushed down others that MiniLM had right, including two sections that fell from first to sixth. A subset chosen because it failed will always favour the challenger. I removed the Qwen3 connector and kept MiniLM.

**The references.** Five reference answers are wrong, or not grounded in the paper, when checked against the PDF. One example: the reference says sequential training gives higher gains than jump training, and Table 1 of the paper shows 7.7 against 12.5. Those five sit in a file with the reason for each. Every report gives the metrics with them and without them.

**The heading tree.** A few sections were split by a heading that is not one: a chart title, a table placeholder, a numbered list item. Those are Docling's, and they lead to the next section.

## 12. Checks that need no reference

A benchmark tells you how the chain did on 233 questions someone else wrote. The documents a user loads come with no question and no reference answer. So I wrote checks for each step of the pipeline that need neither. Each check gives a value, a pass or fail, a rationale and the item it is about, in the shape MLflow expects for a feedback on a trace. Some are plain code, run on every document. Some are LLM judges, run on a sample.

I calibrated them on the 60 papers of the sample.

**Parsing.** I compare the words of the PDF text layer with the words Docling extracted, page by page. The median page keeps 98 % of its words. Four pages out of 1,454 fall under 60 %, and all four are real losses. On three of them, the text of vector charts came out as `\x00` characters. On the fourth, a figure caption was missing altogether.

**Formulas.** A structural check of the LaTeX found 14 papers with at least one broken transcription. Two patterns stood out. Some transcriptions stop right before a `<` sign, which the PDF text still has. Others end in 2,400 to 3,000 characters of `\quad`, a model looping until it ran out of tokens. A vision judge on a 5 % sample compares the LaTeX with the crop of the formula and scores 78 transcriptions out of 81 at 4 or 5.

**Figure descriptions.** The same judge compares each description with its image. It failed 20 of the 21 descriptions it saw. I checked two by hand, and the judge was right both times. One description of a bar chart read "Detailer Match Score" for "Dataflow Match Score", 0.53282 for 0.5282 and "Syntac" for "Syntax". For search, a question about Dataflow will not find that chart. Twenty-one figures is a small sample, but the signal is clear: the descriptions need work before the figures can be trusted.

**The heading tree.** One check follows the numbering of the headings: 2.1, 2.2, 3. It caught list items read as headings ("1. Crossover" after section 2) and headings lost on the way. Another checks that bibliographies and acknowledgements stay out of the index. It found eight papers where they did not. In one, a bibliography of 28 entries sat inside the conclusion: its heading had not been read as a heading. In the others, an "Acknowledgements." paragraph without a heading of its own, or a funding note attached to the introduction, went into the index. A text judge reads the outline of each paper and lists the headings out of place. It flagged chart titles, author names and table placeholders.

**Chunks.** Duplicates, empty chunks, the share of very short chunks, and one Qdrant point per chunk. The duplicates are mostly running headers and affiliations: "manuscript submitted to acm" nine times in one paper.

One lesson came from the judges themselves. Claude Haiku 5.5 thinks by default, and its thinking counts against the token limit of the reply. At 800 tokens, one judge reply in six stopped before or inside its JSON. My benchmark judge, at 600 tokens, had a reply cut once in 233, most likely for the same reason, and I had not noticed. The judges now run with thinking turned off. The answer generation keeps the model's default.

## 13. Limits, stated plainly

- Every metric here is home-made: a 5-gram overlap or a matching heading, an LLM judge from 1 to 5, and checks I calibrated on one sample of 60 papers. They are not Vectara's official metrics.
- The judges are language models. The second opinion corrects some errors of the first and probably adds its own. That is why I report both scores.
- The three questions of sections 5 to 7 were chosen because they failed. They show that the mechanisms work, not that the benchmark improves. The reranker test of section 11 shows how a failing subset can mislead.
- Formula readings add one text-model call per formula at indexing time: about 2,600 calls for this sample.
- Everything runs on a CPU. Stronger embedding models exist and are supported in the code, but I cannot use them here.

## 14. What comes next

- Fix what the checks found: the formula transcriptions that loop or stop at `<`, the figure descriptions, and the headings that come from inside a figure or a list.
- Trace every step of the pipeline in MLflow, with its inputs and outputs, and attach these checks to the traces. A document that fails a check should be visible before anyone asks it a question.
- Look at the chunks themselves. Docling gives one chunk per element, 41 words at the median, and 23 % of the chunks have fewer than 15 words. MiniLM reads 256 tokens and truncates 11 % of the embedded texts, most of the tables among them.

If you have run your own pipeline on Open RAG Bench, I would like to compare numbers, and especially failures.
