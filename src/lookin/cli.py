"""Command-line entry point.

Mirrors the `mlflow ui` pattern: the package installs a `lookin` command,
and `lookin ui` launches the web interface.
"""
from __future__ import annotations

import argparse

from lookin import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lookin",
        description="Lookin -- a visual explorer for deep-learning classification results.",
    )
    parser.add_argument("--version", action="version", version=f"lookin {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Lookin web interface.")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    ui_parser.add_argument("--port", type=int, default=8550, help="Port to bind to (default: 8550).")
    ui_parser.add_argument(
        "--dataset", default=None, metavar="PATH",
        help="Dataset CSV to load automatically on startup, instead of typing/browsing for it in the UI.",
    )
    ui_parser.add_argument(
        "--library", default=None, metavar="PATH",
        help="Directory holding saved workspaces and views (default: ~/.config/lookin/library, or "
             "$LOOKIN_LIBRARY). Point it at a shared mount to open a colleague's workspace by id.",
    )
    ui_parser.add_argument(
        "--experiment", action="append", default=[], metavar="NAME=PATH", dest="experiments",
        help="Results CSV to load automatically on startup, as NAME=PATH. Repeatable for multiple "
             "experiments. Requires --dataset.",
    )

    args = parser.parse_args(argv)

    if args.command == "ui":
        if args.experiments and not args.dataset:
            parser.error("--experiment requires --dataset to also be set.")
        experiments = []
        for spec in args.experiments:
            if "=" not in spec:
                parser.error(f"--experiment must be NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            experiments.append((name, path))

        from lookin.app import run_ui  # deferred: keep `lookin --help` fast and dependency-light
        run_ui(host=args.host, port=args.port, dataset=args.dataset, experiments=experiments,
               library_dir=args.library)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
