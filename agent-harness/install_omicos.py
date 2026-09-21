#!/usr/bin/env python3
"""Copy the portable Agent/Skill bundle; never overwrite an existing installation."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


AGENT = "cytotrace2_fast_analyst.md"
SKILL = "cytotrace2-fast"


def install(destination, layout="workspace"):
    if layout not in {"workspace", "catalog"}:
        raise ValueError("layout must be workspace or catalog")
    source = Path(__file__).resolve().parent / "omicos"
    root = Path(destination).expanduser().resolve()
    if layout == "catalog":
        root = root / "domains" / "biology"
    targets = [(source / "agents" / AGENT, root / "agents" / AGENT),
               (source / "skills" / SKILL, root / "skills" / SKILL)]
    for src, dst in targets:
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(f"refusing to overwrite: {dst}")
    created = []
    try:
        for src, dst in targets:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                # Claim the directory before copying, so rollback never deletes someone else's path.
                dst.mkdir()
                created.append(dst)
                shutil.copytree(src, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                with dst.open("xb") as stream:
                    created.append(dst)
                    stream.write(src.read_bytes())
    except BaseException:
        for path in reversed(created):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        raise
    return [str(dst) for _, dst in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--layout", choices=["workspace", "catalog"], default="workspace")
    opts = parser.parse_args()
    try:
        for path in install(opts.destination, opts.layout):
            print(path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
