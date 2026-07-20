"""Unified CLI for data collection and performance benchmarks.

Thin UX layer over ``sdft.records`` — schemas and persistence live there.
See ``docs/shared-contract.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .records import (
    collect_record,
    collected_records_path,
    export_collected_for_training,
    import_training_row,
    list_performance_results,
    load_collected_records,
    load_performance_result,
    performance_dir,
    performance_result_path,
    run_benchmark,
)


def _cmd_collect(args: argparse.Namespace) -> None:
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise SystemExit(f"file not found: {path}")
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = import_training_row(json.loads(line), source="cli")
            print(f"collected {record.id}")
            n += 1
        print(f"imported {n} row(s) -> {collected_records_path()}")
        return

    if not args.instruction:
        raise SystemExit("provide --instruction or --file")

    tags = list(args.tag or [])
    record = collect_record(
        args.instruction,
        input=args.input or "",
        output=args.output or "",
        source="cli",
        tags=tags,
    )
    print(f"collected {record.id}")
    print(f"  source: {record.source}")
    print(f"  wrote {collected_records_path()}")


def _cmd_export(args: argparse.Namespace) -> None:
    out, count = export_collected_for_training(
        args.name,
        require_output=not args.allow_empty_output,
    )
    print(f"exported {count} training row(s) -> {out}")
    if count:
        print(
            "point a config at this path, e.g. data.data_files with "
            "data.dataset: json in your YAML config"
        )


def _cmd_bench(args: argparse.Namespace) -> None:
    prompts = list(args.prompt) if args.prompt else None
    result = run_benchmark(
        args.benchmark,
        config_path=args.config,
        num_examples=args.num_examples,
        prompts=prompts,
        persist=not args.no_persist,
    )
    m = result.metrics
    out = performance_result_path(result.id)
    print(f"benchmark {result.id} ({result.benchmark})")
    print(f"  model: {result.model}")
    print(f"  device: {m.device}")
    print(f"  samples: {m.samples}")
    print(f"  latency mean: {m.latency_ms_mean:.1f} ms")
    print(f"  tokens/sec: {m.tokens_per_second:.1f}")
    if result.metadata.get("generations_path"):
        print(f"  generations: {result.metadata['generations_path']}")
    if not args.no_persist:
        print(f"  wrote {out}")


def _cmd_list(args: argparse.Namespace) -> None:
    if args.what == "records":
        records = load_collected_records(collected_records_path())
        if not records:
            print("no collected records yet")
            return
        for rec in records[-args.limit :]:
            tags = f" tags={rec.tags}" if rec.tags else ""
            preview = rec.instruction.replace("\n", " ")[:60]
            print(f"{rec.id}  {rec.source}{tags}  {preview}")
        print(f"({len(records)} total in {collected_records_path()})")
        return

    results = list_performance_results(performance_dir())
    if not results:
        print("no benchmark runs yet")
        return
    for res in results[-args.limit :]:
        print(
            f"{res.id}  {res.benchmark}  "
            f"{res.metrics.tokens_per_second:.1f} tok/s  "
            f"{res.metrics.latency_ms_mean:.1f} ms  {res.run_at}"
        )
    print(f"({len(results)} total in {performance_dir()})")


def _cmd_show(args: argparse.Namespace) -> None:
    path = performance_result_path(args.run_id)
    if not path.is_file():
        raise SystemExit(f"benchmark not found: {path}")
    result = load_performance_result(path)
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdft.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m sdft.cli collect "
            '-i "Explain LoRA in one sentence." -o "Low-rank adapters for fine-tuning."\n'
            "  uv run python -m sdft.cli collect --file data/my_dataset.jsonl\n"
            "  uv run python -m sdft.cli export my-batch\n"
            "  uv run python -m sdft.cli bench generate --num-examples 4\n"
            "  uv run python -m sdft.cli bench inference --prompt \"What is LoRA?\"\n"
            "  uv run python -m sdft.cli list records\n"
            "  uv run python -m sdft.cli list benchmarks\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser(
        "collect",
        help="Append a training example (or import JSONL) via sdft.records",
    )
    p_collect.add_argument("-i", "--instruction", default=None, help="User instruction / prompt")
    p_collect.add_argument("--input", default="", help="Optional context field")
    p_collect.add_argument("-o", "--output", default="", help="Expected completion")
    p_collect.add_argument("--tag", action="append", default=None, help="Repeatable tag")
    p_collect.add_argument(
        "--file",
        default=None,
        help="Import Alpaca-style JSONL rows (source=cli)",
    )
    p_collect.set_defaults(func=_cmd_collect)

    p_export = sub.add_parser(
        "export",
        help="Export collected records to training JSONL under data/collected/",
    )
    p_export.add_argument("name", help="Export basename (writes data/collected/<name>.jsonl)")
    p_export.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Include rows with empty output",
    )
    p_export.set_defaults(func=_cmd_export)

    p_bench = sub.add_parser(
        "bench",
        help="Run generate or inference benchmark",
    )
    p_bench.add_argument(
        "benchmark",
        choices=["generate", "inference"],
        help="Benchmark kind",
    )
    p_bench.add_argument("--config", default="configs/default.yaml")
    p_bench.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Examples for generate benchmark (default: 8)",
    )
    p_bench.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Inference prompt (repeatable; defaults to a built-in sample)",
    )
    p_bench.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing outputs/benchmarks/",
    )
    p_bench.set_defaults(func=_cmd_bench)

    p_list = sub.add_parser("list", help="List collected records or benchmark runs")
    p_list.add_argument(
        "what",
        choices=["records", "benchmarks"],
        help="Which index to show",
    )
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print one benchmark result as JSON")
    p_show.add_argument("run_id", help="Benchmark id (e.g. bench-20260719T…)")
    p_show.set_defaults(func=_cmd_show)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
