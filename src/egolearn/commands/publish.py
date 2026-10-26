"""Prepare and publish evaluated checkpoints to the Hub."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egl publish", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="export and validate a local batch")
    prepare_parser.add_argument("manifest", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--allow-legacy-evaluations", action="store_true")
    push_parser = commands.add_parser("push", help="upload models and atomically update results")
    push_parser.add_argument("bundle", type=Path)
    push_parser.add_argument("--results-repo", required=True)
    push_parser.add_argument("--collection")
    push_parser.add_argument("--public-results", action="store_true")
    push_parser.add_argument(
        "--prompt-token", action="store_true", help="read a token without echoing it"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from egolearn.publication.prepare import prepare

            print(prepare(args.manifest, args.output, allow_legacy=args.allow_legacy_evaluations))
        else:
            from egolearn.publication.hub import publish

            if args.prompt_token:
                os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face token: ")
            receipt = publish(
                args.bundle,
                results_repo=args.results_repo,
                collection=args.collection,
                public_results=args.public_results,
            )
            print(json.dumps(receipt, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"egl publish: {error}\n")
    return 0
