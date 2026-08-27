from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="A small, self-contained coding agent.",
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """Run codingAgent from the terminal."""


def main() -> None:
    app(prog_name="coding-agent")
