"""Ask the assistant a question from the terminal.

    python scripts/ask.py "Combien de jours de congés payés par mois ?"
    python scripts/ask.py --demo          # the five questions worth screenshotting
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generator import answer_question  # noqa: E402

BOLD, DIM, GREEN, YELLOW, RED, RESET = ("\033[1m", "\033[2m", "\033[32m",
                                        "\033[33m", "\033[31m", "\033[0m")

DEMO = [
    "Combien de jours de congés payés un salarié acquiert-il par mois de travail ?",
    "Le temps d'habillage est-il payé dans la métallurgie ?",
    "Quel délai pour prévenir un salarié d'une astreinte ?",
    "What is the employer's general safety obligation?",
    "Quelle est la recette de la tarte tatin ?",     # must be refused
]


def show(question: str) -> None:
    print(f"\n{BOLD}? {question}{RESET}")
    t0 = time.perf_counter()
    answer = answer_question(question)
    elapsed = time.perf_counter() - t0

    if not answer.answered:
        # Two very different situations wear the same flag, so name them apart:
        # the guardrail deciding to refuse is the product working, an API
        # failure is the product broken.
        if answer.sources:
            print(f"\n{RED}ERREUR{RESET}  {answer.text or answer.reason}")
        else:
            print(f"\n{YELLOW}REFUSÉ{RESET}  {answer.text or answer.reason}")
            print(f"{DIM}  {elapsed:.1f}s · aucun appel au LLM{RESET}")
            return
    else:
        print(f"\n{answer.text}\n")

    print(f"{DIM}Sources :{RESET}")
    for source in answer.sources:
        mark = f"{GREEN}●{RESET}" if source.cited else f"{DIM}○{RESET}"
        label = source.article or source.document
        print(f"  {mark} [{source.index}] {label}  {DIM}{source.document}{RESET}")
        if source.cited:
            print(f"      {DIM}{source.excerpt[:150].replace(chr(10), ' ')}…{RESET}")
    print(f"{DIM}{elapsed:.1f}s · {answer.model or 'n/a'} · {answer.language}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--demo", action="store_true", help="run the demo set")
    args = parser.parse_args()

    if args.demo:
        for i, question in enumerate(DEMO):
            if i:
                # Space the calls out: free Mistral tiers rate-limit hard, and a
                # burst of five is exactly what trips them.
                time.sleep(2.0)
            show(question)
    elif args.question:
        show(" ".join(args.question))
    else:
        parser.error("give a question, or use --demo")


if __name__ == "__main__":
    main()
