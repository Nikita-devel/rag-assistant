"""Calibrate MIN_RERANK_SCORE from data instead of guessing it.

The guardrail decides when the assistant refuses to answer. It has two ways to
be wrong and they pull in opposite directions:

  false refusal  — a valid question gets "I don't know". The demo looks broken.
  false answer   — an out-of-scope question gets an answer. The demo lies.

The first version of this project set the threshold to 0.0 by hand and only ever
measured the second kind. Result: "le temps d'habillage est-il payé dans la
métallurgie ?" — a question whose answer is Article 96.1, retrieved at rank 1 —
was refused, because the cross-encoder scored it -2.61 and -2.61 < 0.

This script collects the top reranker score for both populations and reports the
threshold that separates them, plus what it costs at every candidate value.

    python scripts/calibrate_guardrail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import Retriever  # noqa: E402

IN_SCOPE = [
    "Combien de jours de congés payés un salarié acquiert-il par mois de travail ?",
    "Quelle est la durée légale du travail hebdomadaire ?",
    "À partir de combien de salariés faut-il mettre en place un CSE ?",
    "Qui doit fournir les équipements de protection individuelle ?",
    "Quelle est l'obligation générale de sécurité de l'employeur ?",
    "Quelle ancienneté pour avoir droit à l'indemnité de licenciement ?",
    "Comment la loi définit-elle le harcèlement moral ?",
    "Le télétravail doit-il être formalisé par un accord ?",
    "Le temps d'habillage est-il payé dans la métallurgie ?",
    "Quel délai pour prévenir un salarié d'une astreinte ?",
    "Comment se calcule l'indemnité de licenciement dans la métallurgie ?",
    "Quelles sont les règles sur le travail de nuit ?",
    "Un CDD peut-il être renouvelé et combien de fois ?",
    "Quel est le taux de majoration des heures supplémentaires ?",
    "Quelles obligations en cas d'exposition au bruit dans l'atelier ?",
    "What is the employer's general safety obligation?",
    "How many paid holiday days per month does an employee earn?",
    "Can a fixed-term contract be renewed?",

    # Written WITHOUT diacritics, the way people actually type on a phone or a
    # non-French keyboard. This is not an edge case: measured on the astreinte
    # question, dropping the accents moved the correct article from rank 1 to
    # rank 5 and the top score from +3.79 to -2.95. A threshold calibrated only
    # on properly accented French is calibrated on a population the demo will
    # not see, and these questions land closest to the refusal boundary — so
    # they are the ones that set it.
    "Quel delai pour prevenir un salarie d une astreinte ?",
    "Combien de jours de conges payes par mois de travail ?",
    "Quelle est la duree legale du travail hebdomadaire ?",
    "Le teletravail doit-il etre formalise par un accord ?",
    "Qui doit fournir les equipements de protection individuelle ?",
    "Comment se calcule l indemnite de licenciement ?",
]

OUT_OF_SCOPE = [
    "Quelle est la recette de la tarte tatin ?",
    "Quelle est la recette de la tarte tatin sans accents ?",
    "Comment configurer un serveur nginx avec TLS ?",
    "Quel est le prix du bitcoin aujourd'hui ?",
    "Quelles sont les règles de la Coupe du monde de handball ?",
    "Qui a gagné la Ligue des champions en 2024 ?",
    "Comment réparer une fuite sous mon évier ?",
    "Quelle est la capitale de l'Australie ?",
    "Écris-moi un poème sur l'automne.",
    "Quels sont les symptômes de la grippe ?",
    "Combien coûte un billet de train Lyon-Paris ?",
]


def top_scores(retriever: Retriever, questions: list[str]) -> list[tuple[str, float]]:
    out = []
    for q in questions:
        res = retriever.search(q, strategy="hybrid_rerank")
        out.append((q, res.top_score if res.hits else float("-inf")))
    return out


def main() -> None:
    retriever = Retriever()

    print(f"scoring {len(IN_SCOPE)} in-scope questions …", flush=True)
    inside = top_scores(retriever, IN_SCOPE)
    print(f"scoring {len(OUT_OF_SCOPE)} out-of-scope questions …", flush=True)
    outside = top_scores(retriever, OUT_OF_SCOPE)

    inside.sort(key=lambda x: x[1])
    outside.sort(key=lambda x: -x[1])

    # French questions typed without diacritics. The English ones are also
    # accent-free but they are a different population, so they are excluded —
    # otherwise the comparison below measures language, not diacritics.
    from app.query_expansion import detect_language
    accentless = {q for q in IN_SCOPE
                  if detect_language(q) == "fr"
                  and not any(c in q for c in "éèêëàâîïôûùçÉÈÊËÀÂÎÏÔÛÙÇ")}

    print("\nIN SCOPE (lowest first — these set the floor)")
    for q, s in inside:
        mark = " [no accents]" if q in accentless else ""
        print(f"  {s:>8.3f}  {q[:52]}{mark}")
    print("\nOUT OF SCOPE (highest first — these set the ceiling)")
    for q, s in outside:
        print(f"  {s:>8.3f}  {q[:66]}")

    lo_in = inside[0][1]
    hi_out = outside[0][1]
    print(f"\nlowest in-scope   {lo_in:.3f}")
    print(f"highest out-scope {hi_out:.3f}")
    print(f"gap               {lo_in - hi_out:+.3f}")

    in_scores = [s for _, s in inside]
    out_scores = [s for _, s in outside]

    print(f"\n{'threshold':>10}{'false refusals':>16}{'false answers':>15}{'errors':>9}")
    print("-" * 50)
    candidates = sorted({round(s, 1) for s in in_scores + out_scores}
                        | {round((lo_in + hi_out) / 2, 1)})
    best, best_cost = None, None
    for t in candidates:
        fr = sum(1 for s in in_scores if s < t)
        fa = sum(1 for s in out_scores if s >= t)
        # A false answer is the worse failure: a demo that invents an answer
        # about labour law destroys the trust the whole product sells. Weight it.
        cost = fr + 2 * fa
        flag = ""
        if best_cost is None or cost < best_cost:
            best, best_cost, flag = t, cost, ""
        print(f"{t:>10.1f}{fr:>12}/{len(in_scores):<3}{fa:>11}/{len(out_scores):<3}{cost:>9}{flag}")

    plain = [s for q, s in inside if q in accentless]
    fancy = [s for q, s in inside if q not in accentless]
    if plain and fancy:
        print(f"\naccented questions   min {min(fancy):>7.3f}  avg {sum(fancy)/len(fancy):>7.3f}")
        print(f"unaccented questions min {min(plain):>7.3f}  avg {sum(plain)/len(plain):>7.3f}")
        print(f"cost of dropping diacritics: {sum(fancy)/len(fancy) - sum(plain)/len(plain):+.2f} "
              f"on the average top score")

    print(f"\nrecommended  MIN_RERANK_SCORE={best:.1f}")
    print("(cost = false refusals + 2x false answers — answering out of scope is")
    print(" the failure that destroys trust, so it is weighted double)")
    print(f"\nSet it in .env:  MIN_RERANK_SCORE={best:.1f}")


if __name__ == "__main__":
    main()
