from __future__ import annotations

import argparse
from pathlib import Path

from weir.cli.utils import guarded_main
from weir.core.application.export import run_export


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a trained algorithm policy to a standalone ONNX file."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the algorithm checkpoint."
    )
    parser.add_argument("--output", default="policy.onnx", help="Output file path.")

    def run(args: argparse.Namespace) -> dict[str, object]:
        output_path = run_export(args.checkpoint, args.output)
        return {"checkpoint": str(args.checkpoint), "output": str(output_path)}

    return guarded_main(parser, run, "export")


if __name__ == "__main__":
    raise SystemExit(main())
