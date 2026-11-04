from __future__ import annotations

import argparse
import sys

from babyinfoquick.age import ALL_BANDS, AgeBand
from babyinfoquick.config import list_configs, load_config, settings


def cmd_fetch(args: argparse.Namespace) -> int:
    from babyinfoquick.corpus.fetch import fetch_all

    fetched, failures = fetch_all(force=args.force)
    print(f"\nfetched {len(fetched)} documents, {len(failures)} failures")
    return 1 if failures or not fetched else 0


def cmd_parse(args: argparse.Namespace) -> int:
    from babyinfoquick.corpus.fetch import load_manifest
    from babyinfoquick.corpus.parse import parse_all

    parsed = parse_all(load_manifest(), allow_ocr=not args.no_ocr)
    print(f"\nparsed {len(parsed)} documents")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from babyinfoquick.index.build import build_index

    config = load_config(args.config)
    total = build_index(config, reuse_chunks=args.reuse_chunks)
    print(f"\nindexed {total} chunks for config {config.name}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from babyinfoquick.service import ask

    bands = [AgeBand(b) for b in args.age] if args.age else None
    answer, result = ask(args.question, args.config, bands)

    print(f"\nconfig       {answer.config_name}")
    print(f"age bands    {[b.value for b in answer.age_bands] or 'none'} ({result.age_source})")
    print(f"retrieval    {answer.retrieval_ms:.0f} ms   total {answer.total_ms:.0f} ms")
    print(f"tokens       in={answer.input_tokens} out={answer.output_tokens}")
    print("\n" + answer.text + "\n")

    for citation in answer.citations:
        print(f"  [{citation.marker}] {citation.title} ({citation.publisher}) p.{citation.page_start}")
        print(f"      {citation.url}")

    if args.show_chunks or answer.retrieved_passages:
        print("\nretrieved chunks:")
        for index, scored in enumerate(result.results, start=1):
            print(f"  [{index}] {scored.chunk.chunk_id}  score={scored.score:.4f}")
            print(f"      section: {scored.chunk.section}")
            print(f"      {scored.chunk.text[:160]}...")
            print(f"      {scored.chunk.title}, PDF p.{scored.chunk.page_start}")
            print(f"      {scored.chunk.url}#page={scored.chunk.page_start}")

    print("\n" + answer.disclaimer)
    return 0


def cmd_configs(args: argparse.Namespace) -> int:
    for name in list_configs():
        config = load_config(name)
        print(f"{name:32s} {config.description}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("babyinfoquick.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.check_labels:
        from babyinfoquick.evaluation.labels import load_labels, validate_passages

        labels = load_labels()
        validate_passages(labels)
        print(f"Validated source references for {len(labels)} queries.")
        return 0
    if args.judge and not (settings.anthropic_api_key or settings.openai_api_key):
        print("--judge requires OPENAI_API_KEY or ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    from babyinfoquick.evaluation.runner import run_all, run_config

    if args.config:
        run_config(args.config, judge=args.judge)
    else:
        run_all(judge=args.judge)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="biq", description="BabyInfoQuick")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the source PDFs")
    p.add_argument("--force", action="store_true", help="re-download cached files")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("parse", help="extract text from the downloaded PDFs")
    p.add_argument("--no-ocr", action="store_true", help="skip the OCR fallback")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("index", help="chunk, embed, and index the corpus")
    p.add_argument("--config", default=settings.default_config)
    p.add_argument("--reuse-chunks", action="store_true", help="reuse chunks on disk")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--config", default=settings.default_config)
    p.add_argument("--age", action="append", choices=[b.value for b in ALL_BANDS])
    p.add_argument("--show-chunks", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("configs", help="list the retrieval configs")
    p.set_defaults(func=cmd_configs)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run the evaluation harness")
    p.add_argument("--config", help="one config, or all configs if omitted")
    p.add_argument("--judge", action="store_true", help="also score answer quality with an LLM")
    p.add_argument("--check-labels", action="store_true", help="verify labels against parsed PDFs")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
