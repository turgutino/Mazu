"""Shared machine-readable output helpers for the CLI's --json commands. One small
module owns the JSON envelope format so every command routes through the same
serialization path instead of five separate ad-hoc json.dumps() calls drifting apart.
"""

import json
import sqlite3

import click

from mazu import __version__

# Bumped only when the envelope shape itself changes in a breaking way (a key
# renamed/removed/retyped) -- adding a new command's --json support, or a new field
# on an existing row shape, is additive and does not require a bump.
SCHEMA_VERSION = 1


def row_to_dict(row: sqlite3.Row, bool_fields: tuple[str, ...] = ()) -> dict:
    """sqlite3.Row -> plain dict, casting the named INTEGER-as-bool columns (SQLite
    has no native boolean type) to real bool so they serialize as true/false, not 0/1.
    """
    d = dict(row)
    for f in bool_fields:
        if f in d and d[f] is not None:
            d[f] = bool(d[f])
    return d


def emit_json(data) -> None:
    """Wraps `data` in the versioned envelope and prints it -- the only place that
    calls click.echo(json.dumps(...)). Every command's --json branch calls this and
    returns; nothing else touches json.dumps directly.
    """
    payload = {"schema_version": SCHEMA_VERSION, "mazu_version": __version__, "data": data}
    click.echo(json.dumps(payload, indent=2, default=str))
