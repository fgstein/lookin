"""Command-line entry point.

Mirrors the `mlflow ui` pattern: the package installs a `lookin` command,
and `lookin web` launches the browser interface.
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

    web_parser = subparsers.add_parser("web", help="Launch the browser interface.")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    web_parser.add_argument("--port", type=int, default=8560, help="Port to bind to (default: 8560).")
    web_parser.add_argument("--dataset", default=None, metavar="PATH",
                            help="Dataset CSV to load automatically in every new tab.")
    web_parser.add_argument(
        "--experiment", action="append", default=[], metavar="NAME=PATH", dest="experiments",
        help="Results CSV to load automatically on startup, as NAME=PATH. Repeatable for multiple "
             "experiments. Requires --dataset.",
    )
    web_parser.add_argument(
        "--library", default=None, metavar="PATH",
        help="Directory holding saved workspaces and views (default: ~/.config/lookin/library, or "
             "$LOOKIN_LIBRARY). Point it at a shared mount to open a colleague's workspace by id.",
    )
    web_parser.add_argument(
        "--public-url", default=None, metavar="URL",
        help="The address people open Lookin at, when it is not the one it binds (a port forward, a "
             "gateway): links handed out by the MCP endpoint start with it.",
    )
    web_parser.add_argument(
        "--no-mcp", action="store_false", dest="mcp",
        help="Do not serve the MCP endpoint for AI agents (served at /mcp on the same port by default).",
    )
    web_parser.add_argument(
        "--mcp-read-only", action="store_true",
        help="Serve only the MCP tools that read: no workspace, snapshot, review decision or file "
             "is created, changed or deleted by an agent.",
    )

    args = parser.parse_args(argv)

    if args.command == "web":
        if args.experiments and not args.dataset:
            parser.error("--experiment requires --dataset to also be set.")
        experiments = []
        for spec in args.experiments:
            if "=" not in spec:
                parser.error(f"--experiment must be NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            experiments.append((name, path))

        from lookin.web.server import run_web  # deferred: keep `lookin --help` fast and dependency-light
        run_web(host=args.host, port=args.port, dataset=args.dataset, experiments=experiments,
                library_dir=args.library, enable_mcp=args.mcp, mcp_read_only=args.mcp_read_only,
                public_url=args.public_url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
