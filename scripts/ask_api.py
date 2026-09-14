"""Query the running API from the terminal, with the stream rendered readably.

    python scripts/ask_api.py "Quel delai pour prevenir un salarie ?"
    python scripts/ask_api.py --sources-only "..."     # just the retrieval
    python scripts/ask_api.py --no-stream "..."        # plain POST /query

Exists because hand-rolled curl one-liners kept breaking on shell quoting —
apostrophes and accents in French questions are exactly what a shell mangles.
Here the question is one argument and nothing reinterprets it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

BOLD, DIM, GREEN, YELLOW, RED, RESET = ("\033[1m", "\033[2m", "\033[32m",
                                        "\033[33m", "\033[31m", "\033[0m")


def base_url() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    host, port = "127.0.0.1", "8077"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("APP_HOST="):
                host = line.split("=", 1)[1].strip() or host
            elif line.startswith("APP_PORT="):
                port = line.split("=", 1)[1].strip() or port
    return f"http://{host}:{port}"


def show_sources(sources: list[dict], cited: set[int] | None = None) -> None:
    print(f"\n{DIM}Sources :{RESET}")
    for s in sources:
        mark = f"{GREEN}●{RESET}" if cited and s["n"] in cited else f"{DIM}○{RESET}"
        print(f"  {mark} [{s['n']}] {s['article'] or s['document']:<22} "
              f"{DIM}{s['score']:>8.3f}  {s['document'][:44]}{RESET}")


def stream(url: str, question: str, sources_only: bool) -> None:
    t0 = time.perf_counter()
    response = requests.post(f"{url}/query/stream", json={"question": question},
                             stream=True, timeout=600)
    response.raise_for_status()

    event, sources, first_token = None, [], None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None or raw == "":
            continue
        if raw.startswith("event: "):
            event = raw[7:]
            continue
        if not raw.startswith("data: "):
            continue
        data = json.loads(raw[6:])

        if event == "sources":
            sources = data["sources"]
            print(f"{DIM}retrieval {time.perf_counter() - t0:.1f}s · "
                  f"{data['model']} · {data['language']}{RESET}")
            show_sources(sources)
            if sources_only:
                return
            print()
        elif event == "token":
            if first_token is None:
                first_token = time.perf_counter() - t0
            print(data["text"], end="", flush=True)
        elif event == "refused":
            print(f"\n{YELLOW}REFUSÉ{RESET}  {data['answer']}")
            print(f"{DIM}  {data.get('reason', '')}{RESET}")
        elif event == "error":
            print(f"\n{RED}ERREUR{RESET}  {data['answer']}")
            print(f"{DIM}  {data.get('reason', '')}{RESET}")
        elif event == "final":
            cited = set(data["cited"])
            show_sources(sources, cited)
            ttft = f"{first_token:.1f}s" if first_token else "n/a"
            print(f"{DIM}first token {ttft} · total {data['elapsed_seconds']}s{RESET}")


def blocking(url: str, question: str) -> None:
    response = requests.post(f"{url}/query", json={"question": question}, timeout=600)
    response.raise_for_status()
    data = response.json()
    print(f"\n{data['answer']}")
    show_sources(data["sources"], {s["n"] for s in data["sources"] if s["cited"]})
    print(f"{DIM}{data['elapsed_seconds']}s · {data['model']} · "
          f"{data['language']}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--sources-only", action="store_true",
                        help="stop after retrieval; no model call is waited on")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--url", default=None)
    args = parser.parse_args()

    url = args.url or base_url()
    question = " ".join(args.question)
    print(f"{BOLD}? {question}{RESET}  {DIM}-> {url}{RESET}")

    try:
        if args.no_stream:
            blocking(url, question)
        else:
            stream(url, question, args.sources_only)
    except requests.exceptions.ConnectionError:
        sys.exit(f"\ncannot reach {url} — is the server running? "
                 f"(bash scripts/serve.sh --dev)")


if __name__ == "__main__":
    main()
