from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from weir.cli.utils import guarded_main
from weir.core.application.export import run_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-export",
        description="Export a trained algorithm policy to a standalone ONNX file.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the algorithm checkpoint."
    )
    parser.add_argument("--output", default="policy.onnx", help="Output file path.")
    return parser


def _run_export_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_path = run_export(args.checkpoint, Path(args.output))
    return {"checkpoint": str(args.checkpoint), "output": str(output_path)}


def _print_output(result: dict[str, Any]) -> None:
    print(f"Exported policy to {result['output']}")


def main() -> int:
    return guarded_main(
        build_parser(),
        _run_export_from_args,
        "export",
        on_success=_print_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
