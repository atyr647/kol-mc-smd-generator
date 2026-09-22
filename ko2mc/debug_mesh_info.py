"""Batch .n3pmesh info tool.

Usage:
    python -m ko2mc.debug_mesh_info <file_or_dir> [...]
    python -m ko2mc.debug_mesh_info path/to/Object/          # all .n3pmesh in dir
    python -m ko2mc.debug_mesh_info path/to/Object/ -r       # recurse subdirs
    python -m ko2mc.debug_mesh_info a.n3pmesh b.n3pmesh      # specific files
    python -m ko2mc.debug_mesh_info path/to/Object/ --json out.json
"""

import argparse
import glob as _glob
import json
import sys
from pathlib import Path
from typing import Optional

from .binary_reader import BinaryParseError
from .debug_export import mesh_summary_json, print_mesh_stats
from .n3pmesh_parser import parse_n3pmesh


def _process_file(path: Path, quiet: bool) -> Optional[dict]:
    try:
        mesh = parse_n3pmesh(path)
    except BinaryParseError as e:
        print(f"  PARSE ERROR [{path.name}]: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR [{path.name}]: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    if not quiet:
        print(f"\n[{path.name}]")
        print_mesh_stats(mesh)

    return mesh_summary_json(mesh)


def _collect_files(paths: list[str], recursive: bool) -> list[Path]:
    files = []
    for pat in paths:
        p = Path(pat)
        if p.is_dir():
            pattern = "**/*.n3pmesh" if recursive else "*.n3pmesh"
            files.extend(sorted(p.glob(pattern)))
        else:
            expanded = sorted(Path(g) for g in _glob.glob(pat, recursive=recursive))
            if expanded:
                files.extend(expanded)
            elif p.exists():
                files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Batch .n3pmesh info tool")
    parser.add_argument("paths", nargs="+", help="Files, directories, or glob patterns")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress per-file output (summary only)"
    )
    parser.add_argument("--json", metavar="FILE", help="Write JSON summary array to file")
    args = parser.parse_args()

    files = _collect_files(args.paths, args.recursive)
    if not files:
        print("No .n3pmesh files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} files...")

    results = []
    ok = err = 0
    for f in files:
        summary = _process_file(f, quiet=args.quiet)
        if summary is not None:
            results.append(summary)
            ok += 1
        else:
            err += 1

    print(f"\n--- {ok} OK, {err} errors out of {len(files)} files ---")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)
        print(f"JSON summary written: {args.json}")


if __name__ == "__main__":
    main()
