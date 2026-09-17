# An assistant that cites its sources — and knows when to stop

> Case study · document retrieval assistant for an industrial SME
> **Live demo:** https://assistant-droit-travail-405627799203.europe-west9.run.app
> **Web version of this case study:** https://claude.ai/artifact/CUxT52d6B6VDNeKQ38XVyS

A retrieval assistant over the French Labour Code and the metallurgy collective
agreement (IDCC 3248), built for a sheet-metal company. Every claim points at the
exact article. When the answer is not in the corpus, the language model is never
called at all.

2,204 articles · 2,634 indexed extracts · 3–4 seconds per answer ·
Python, FastAPI, Chroma.

## The problem

A small industrial company accumulates rules it never reads: the Labour Code, a
collective agreement of several hundred articles, internal procedures. Every week
somebody spends an hour looking up what the rule actually says about overtime,
on-call duty or dressing time.

A chatbot answering from memory is worse than useless here: a confidently wrong
answer about working time costs real money and surfaces in a labour tribunal. So
the goal was never "sounds knowledgeable". It was **verifiable**.

## Two design constraints

**Every claim carries its article.** Extracts arrive numbered, the model must
cite by number, and those numbers are resolved back into real references. The
reader does not have to trust the system — they click and read Article L3141-3
themselves.

**No context, no model call.** When retrieval is not confident, the language
model is not invoked at all — not invoked with a careful prompt asking it to
refuse.

> A prompt instruction is a request. Skipping the call is a guarantee.

## What was measured

13 questions whose correct articles were verified one by one against the built
corpus, plus 4 out-of-scope questions that must be refused.

| Strategy | Recall@1 | Recall@5 | MRR | Correct refusals |
|---|---|---|---|---|
| Dense only (e5-small) | 0.23 | 0.54 | 0.34 | 0 / 4 |
| Lexical only (BM25) | 0.15 | 0.62 | 0.35 | 1 / 4 |
| Hybrid (RRF fusion) | 0.31 | 0.62 | 0.44 | 0 / 4 |
| **Hybrid + cross-encoder reranking** | **0.54** | **0.85** | **0.60** | **4 / 4** |

Dense search alone finds the article 54% of the time. Legal text is thick with
near-identical boilerplate, and a bi-encoder maps most of it into a narrow cone.
Lexical search catches the exact terms — *astreinte*, *habillage*, an article
number — but misses paraphrase. Fusing both and then reranking the candidates
with a cross-encoder takes it to 85%.

The last column is the one demos usually leave out. The refusal threshold is not
hand-picked: it is calibrated on data, weighting a wrong answer twice as heavily
as a wrong refusal.

## What had to break first

**1. The evaluation was broken before the system was.** The first version
compared "Article L3141-3" against the label "L3141-3" and scored every strategy
at 0.00 — a measurement bug that looked exactly like total failure. A benchmark
you have not tested is not a benchmark.

**2. Retrieval was not deterministic.** The vector store promises no read order,
that order became the lexical index, and the index became the tie-break order
everywhere downstream. Two runs over an identical index produced different scores
— and a different calibrated threshold. Sorting by content hash fixed it; a test
keeps it fixed.

**3. The refusal threshold was a guess.** Set by hand, it refused perfectly valid
questions: "is dressing time paid" scored −2.61 and was refused. Only one kind of
error was being measured. The calibration tool now measures both.

**4. Questions typed without accents broke retrieval.** Asked "Quel delai pour
prevenir un salarie", the correct article fell from rank 1 to rank 5: lexical
search folds accents on both sides, while the embedder sees *delai* and *délai*
as different words. The restoration table is generated from the corpus itself and
keeps only unambiguous mappings — which is what keeps *sur*/*sûr* and
*cote*/*côte* out of it.

## What this means for your company

This public demo runs on cloud infrastructure, over public legal texts. For
internal documents — your procedures, your company agreements, your technical
sheets — the same system installs on your own hardware: no data leaves the
building. That is one configuration line, not a rewrite.

I am a freelance developer and I build this kind of internal tool for industrial
SMEs.

---

Corpus: French government open data published by DILA (Labour Code; metallurgy
collective agreement, IDCC 3248). Technical demonstration — not legal advice.
Source and benchmark: https://github.com/Nikita-devel/rag-assistant
