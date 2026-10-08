"""Command line for the positive-valence steering sandbox."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console

from pvsteer.config import load_config
from pvsteer.pipeline import (
    behavior_command,
    extract_activations_command,
    fit_directions_command,
    make_dataset,
    report_command,
    run_all,
    steer_command,
    unembedding_command,
    validate_command,
)
from pvsteer.safety import SafetyError

COMMANDS = {
    "make-dataset": make_dataset,
    "extract-activations": extract_activations_command,
    "fit-directions": fit_directions_command,
    "validate": validate_command,
    "unembedding": unembedding_command,
    "steer-generate": steer_command,
    "behavioral-benchmark": behavior_command,
    "report": report_command,
    "run-all": run_all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvsteer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--device", default=None)
        command.add_argument("--model-name", default=None)
        command.add_argument("--output-dir", default=None)
        command.add_argument("--safe-mode", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    console = Console(stderr=True)
    args = build_parser().parse_args(argv)
    overrides = {
        "seed": args.seed,
        "device": args.device,
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "safe_mode": args.safe_mode,
    }
    try:
        config = load_config(args.config, overrides)
    except (SafetyError, ValueError) as exc:
        console.print(f"configuration rejected: {exc}")
        return 2
    try:
        result = COMMANDS[args.command](config)
    except SafetyError as exc:
        console.print(f"refused: {exc}")
        return 2
    except RuntimeError as exc:
        console.print(str(exc))
        return 1
    console.print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
