"""Measure retrieval quality against labelled ground truth.

Each question is paired with the set of articles that legitimately answer it —
verified by grepping the built corpus. A set, not a single string: in a legal
corpus several articles often answer the same question (the Code and the
convention both cover working time; L1251-23 covers PPE for temp workers while
R4323-95 covers it generally). Scoring against one arbitrary "right" article
measures the labeller's taste, not the retriever.

Sub-articles count: if 75.3 is accepted, 75.3.3 is a hit. The convention nests
its numbering, and a chunk from 75.3.3 does answer a question about 75.3.

Metrics:
  Recall@1  — correct article ranked first
  Recall@5  — correct article anywhere in the top 5
  MRR       — 1/rank, so ranking it first is worth more than ranking it fifth
  Rejection — on out-of-corpus questions, does the guardrail correctly refuse

Usage:
    python scripts/eval_retrieval.py --show-misses
    python scripts/eval_retrieval.py --html report.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import Retriever  # noqa: E402

# (question, {articles that legitimately answer it})
LABELLED: list[tuple[str, set[str]]] = [
    ("Combien de jours de congés payés par mois de travail effectif ?",
     {"L3141-3"}),
    ("Quelle est la durée légale du travail hebdomadaire ?",
     {"L3121-27", "95"}),
    ("À partir de combien de salariés faut-il mettre en place un CSE ?",
     {"L2311-2"}),
    ("Qui doit fournir les équipements de protection individuelle et qui les paie ?",
     {"R4323-95", "R4323-91", "L4122-2", "L1251-23"}),
    ("Quelle est l'obligation générale de sécurité de l'employeur ?",
     {"L4121-1", "L4121-2"}),
    ("Quelle ancienneté faut-il pour avoir droit à l'indemnité de licenciement ?",
     {"L1234-9"}),
    ("Comment la loi définit-elle le harcèlement moral ?",
     {"L1152-1"}),
    ("Le télétravail doit-il être formalisé par un accord ?",
     {"L1222-9"}),
    ("Le temps d'habillage est-il du temps de travail effectif dans la métallurgie ?",
     {"96.1", "L3121-3"}),
    ("Quel délai de prévenance pour informer un salarié de son programme d'astreinte ?",
     {"96.2.1.2", "L3121-12"}),
    ("Comment se calcule l'indemnité de licenciement dans la convention métallurgie ?",
     {"75.3"}),
    ("How many paid holiday days does an employee accrue per month worked?",
     {"L3141-3"}),
    ("What is the employer's general safety obligation?",
     {"L4121-1", "L4121-2"}),
]

OUT_OF_CORPUS: list[str] = [
    "Quelle est la recette de la tarte tatin ?",
    "Comment configurer un serveur nginx avec TLS ?",
    "Quel est le prix du bitcoin aujourd'hui ?",
    "Quelles sont les règles de la Coupe du monde de handball ?",
]

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
K = 5

# Chunk metadata stores "Article L3141-3"; labels are written bare. Normalise
# both sides — the first version of this script compared the two raw and scored
# every strategy at 0.00.
ARTICLE_PREFIX = re.compile(r"^\s*art(?:icle|\.)?\s*", re.IGNORECASE)


def normalize(article: str) -> str:
    article = ARTICLE_PREFIX.sub("", article or "")
    return re.sub(r"\s+", "", article).upper().rstrip(".")


def matches(found: str, accepted: set[str]) -> bool:
    found = normalize(found)
    if not found:
        return False
    for target in accepted:
        target = normalize(target)
        if found == target or found.startswith(f"{target}."):
            return True
    return False


def rank_of(hits, accepted: set[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if matches(hit.metadata.get("article", ""), accepted):
            return rank
    return None


def evaluate(retriever: Retriever, strategy: str) -> dict:
    per_question = []
    recall_1 = recall_k = 0
    rr_total = 0.0
    t0 = time.perf_counter()

    for question, accepted in LABELLED:
        res = retriever.search(question, top_k=K, strategy=strategy)
        rank = rank_of(res.hits, accepted)
        top = res.hits[0].metadata.get("article", "—") if res.hits else "—"
        if rank == 1:
            recall_1 += 1
        if rank is not None:
            recall_k += 1
            rr_total += 1.0 / rank
        per_question.append({
            "question": question,
            "expected": sorted(accepted),
            "rank": rank,
            "top": normalize(top) or "—",
        })

    elapsed = (time.perf_counter() - t0) / len(LABELLED)

    rejections = []
    for q in OUT_OF_CORPUS:
        res = retriever.search(q, top_k=K, strategy=strategy)
        rejections.append({"question": q, "rejected": not res.confident,
                           "top_score": round(res.top_score, 3)})

    n = len(LABELLED)
    return {
        "strategy": strategy,
        "recall_1": recall_1 / n,
        "recall_k": recall_k / n,
        "mrr": rr_total / n,
        "rejected": sum(r["rejected"] for r in rejections),
        "reject_total": len(OUT_OF_CORPUS),
        "sec_per_query": elapsed,
        "per_question": per_question,
        "rejections": rejections,
    }


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def render_html(results: list[dict]) -> str:
    e = html.escape
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = max(results, key=lambda r: (r["recall_1"], r["mrr"]))

    def bar_rows(metric: str, fmt: str = "{:.0%}") -> str:
        peak = max(r[metric] for r in results) or 1.0
        out = []
        for r in results:
            width = r[metric] / peak * 100
            out.append(
                f'<div class="bar-row"><span class="bar-label">{e(r["strategy"])}</span>'
                f'<span class="bar-track"><span class="bar-fill" style="width:{width:.1f}%"></span></span>'
                f'<span class="bar-value">{fmt.format(r[metric])}</span></div>'
            )
        return "".join(out)

    # Latency is the one metric where lower is better, so it gets plain values
    # rather than bars — a long bar next to three "longer is better" charts
    # reads as a win when it is the opposite.
    lat_rows = "".join(
        f'<div class="bar-row lat"><span class="bar-label">{e(r["strategy"])}</span>'
        f'<span class="bar-value">{r["sec_per_query"]:.2f}s</span></div>'
        for r in results
    )

    questions = [q["question"] for q in results[0]["per_question"]]
    grid_head = "".join(f"<th>{e(r['strategy'])}</th>" for r in results)
    grid_rows = []
    for i, question in enumerate(questions):
        cells = []
        for r in results:
            item = r["per_question"][i]
            rank = item["rank"]
            if rank == 1:
                cls, label, note = "hit", "1", "rank 1"
            elif rank:
                cls, label, note = "partial", str(rank), f"rank {rank}"
            else:
                cls, label, note = "miss", "—", f"got {item['top']}"
            cells.append(f'<td class="cell {cls}" title="{e(note)}">'
                         f'<span class="dot"></span>{label}</td>')
        expected = ", ".join(results[0]["per_question"][i]["expected"])
        grid_rows.append(
            f'<tr><th scope="row"><span class="q">{e(question)}</span>'
            f'<span class="exp">{e(expected)}</span></th>{"".join(cells)}</tr>'
        )

    reject_rows = []
    for i, rej in enumerate(results[0]["rejections"]):
        cells = []
        for r in results:
            ok = r["rejections"][i]["rejected"]
            cells.append(f'<td class="cell {"hit" if ok else "miss"}">'
                         f'<span class="dot"></span>{"refusé" if ok else "répondu"}</td>')
        reject_rows.append(
            f'<tr><th scope="row"><span class="q">{e(rej["question"])}</span></th>'
            f'{"".join(cells)}</tr>')

    table_json = e(json.dumps(
        [{k: v for k, v in r.items() if k not in ("per_question", "rejections")}
         for r in results], ensure_ascii=False, indent=2))

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval evaluation — RAG assistant</title>
<style>
:root {{
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10);
  --accent:#2a78d6; --good:#0ca30c; --warn:#fab219; --bad:#d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10);
    --accent:#3987e5;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10);
  --accent:#3987e5;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding-block:32px; padding-inline:16px; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:15px; margin:0 0 14px; color:var(--ink-2); font-weight:600; }}
.sub {{ color:var(--muted); margin:0 0 28px; font-size:13px; }}
.card {{ background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:20px; margin-bottom:20px; }}
.hero {{ display:flex; gap:32px; flex-wrap:wrap; align-items:flex-end;
  margin-bottom:20px; }}
.hero .num {{ font-size:40px; font-weight:650; line-height:1; letter-spacing:-0.02em; }}
.hero .cap {{ color:var(--muted); font-size:12px; margin-top:6px;
  text-transform:uppercase; letter-spacing:0.05em; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:24px; }}
.metric h3 {{ font-size:12px; margin:0 0 10px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }}
.bar-row {{ display:grid; grid-template-columns:96px 1fr 46px;
  align-items:center; gap:8px; margin-bottom:6px; }}
.bar-label {{ color:var(--ink-2); font-size:12px; }}
.bar-track {{ height:9px; background:var(--grid); border-radius:4px; }}
.bar-fill {{ display:block; height:100%; background:var(--accent);
  border-radius:0 4px 4px 0; min-width:2px; }}
.bar-row.lat {{ grid-template-columns:96px 1fr; }}
.bar-row.lat .bar-value {{ text-align:left; }}
.bar-value {{ text-align:right; font-size:12px; color:var(--ink-2);
  font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--grid); }}
thead th {{ color:var(--muted); font-size:11px; text-transform:uppercase;
  letter-spacing:0.05em; font-weight:600; }}
tbody th {{ font-weight:400; max-width:400px; }}
.q {{ display:block; color:var(--ink); }}
.exp {{ display:block; color:var(--muted); font-size:11px;
  font-variant-numeric:tabular-nums; margin-top:2px; }}
.cell {{ font-variant-numeric:tabular-nums; white-space:nowrap; font-size:12px;
  color:var(--ink-2); }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:7px; vertical-align:-1px; }}
.hit .dot {{ background:var(--good); }}
.partial .dot {{ background:var(--warn); }}
.miss .dot {{ background:var(--bad); }}
details {{ margin-top:8px; }}
summary {{ cursor:pointer; color:var(--ink-2); font-size:13px; }}
pre {{ background:var(--plane); border:1px solid var(--border); border-radius:8px;
  padding:14px; overflow-x:auto; font-size:12px; color:var(--ink-2); }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:12px;
  font-size:12px; color:var(--muted); }}
</style></head><body><div class="wrap">

<h1>Évaluation du retrieval</h1>
<p class="sub">Corpus : Code du travail + convention collective métallurgie (IDCC 3248)
 · {len(questions)} questions étiquetées · {len(OUT_OF_CORPUS)} questions hors corpus
 · {e(stamp)}</p>

<div class="card">
  <div class="hero">
    <div><div class="num">{best['recall_1']:.0%}</div>
      <div class="cap">Recall@1 — {e(best['strategy'])}</div></div>
    <div><div class="num">{best['rejected']}/{best['reject_total']}</div>
      <div class="cap">refus corrects hors corpus</div></div>
    <div><div class="num">{best['sec_per_query']:.2f}s</div>
      <div class="cap">par requête (CPU)</div></div>
  </div>
  <div class="metrics">
    <div class="metric"><h3>Recall@1</h3>{bar_rows("recall_1")}</div>
    <div class="metric"><h3>Recall@{K}</h3>{bar_rows("recall_k")}</div>
    <div class="metric"><h3>MRR</h3>{bar_rows("mrr", "{:.2f}")}</div>
    <div class="metric"><h3>Latence (s/requête)</h3>{lat_rows}</div>
  </div>
</div>

<div class="card">
  <h2>Par question — rang de l'article attendu</h2>
  <div class="scroll"><table>
    <thead><tr><th>Question / articles acceptés</th>{grid_head}</tr></thead>
    <tbody>{"".join(grid_rows)}</tbody>
  </table></div>
  <div class="legend">
    <span><span class="dot" style="background:var(--good)"></span>rang 1</span>
    <span><span class="dot" style="background:var(--warn)"></span>rang 2–{K}</span>
    <span><span class="dot" style="background:var(--bad)"></span>absent du top {K}</span>
  </div>
</div>

<div class="card">
  <h2>Garde-fou — questions hors corpus</h2>
  <div class="scroll"><table>
    <thead><tr><th>Question</th>{grid_head}</tr></thead>
    <tbody>{"".join(reject_rows)}</tbody>
  </table></div>
</div>

<div class="card">
  <h2>Données brutes</h2>
  <details><summary>Afficher le JSON</summary><pre>{table_json}</pre></details>
</div>

</div></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--html", type=Path, help="write an HTML report here")
    args = parser.parse_args()

    retriever = Retriever()
    strategies = [args.strategy] if args.strategy else STRATEGIES

    results = []
    for strategy in strategies:
        print(f"evaluating {strategy} …", flush=True)
        results.append(evaluate(retriever, strategy))

    header = (f"{'strategy':<16}{'R@1':>7}{f'R@{K}':>7}{'MRR':>7}"
              f"{'reject':>9}{'s/query':>10}")
    print(f"\n{header}\n{'-' * len(header)}")
    for r in results:
        reject = f"{r['rejected']}/{r['reject_total']}"
        print(f"{r['strategy']:<16}{r['recall_1']:>7.2f}{r['recall_k']:>7.2f}"
              f"{r['mrr']:>7.2f}{reject:>9}{r['sec_per_query']:>10.2f}")

    if args.show_misses:
        for r in results:
            misses = [q for q in r["per_question"] if q["rank"] is None]
            if misses:
                print(f"\n--- {r['strategy']} misses ---")
                for q in misses:
                    print(f"  expected {'/'.join(q['expected']):<22} "
                          f"got {q['top']:<16} | {q['question'][:56]}")

    if args.html:
        args.html.write_text(render_html(results), encoding="utf-8")
        print(f"\nHTML report -> {args.html}")


if __name__ == "__main__":
    main()
