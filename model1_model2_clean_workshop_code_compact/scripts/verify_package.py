#!/usr/bin/env python3
"""Verify the file manifest and the selected checkpoint hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ignored_generated_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        relative.parts[0] == "runs"
        or path.name == ".DS_Store"
        or "__pycache__" in relative.parts
        or ".pytest_cache" in relative.parts
        or "sbi-logs" in relative.parts
        or path.suffix in {".pyc", ".pyo", ".log"}
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_hashes(node: object) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_path") or key == "path":
                hash_key = key[:-5] + "_sha256" if key.endswith("_path") else "sha256"
                if hash_key in node:
                    output.append((str(value), str(node[hash_key])))
            output.extend(selected_hashes(value))
    elif isinstance(node, list):
        for value in node:
            output.extend(selected_hashes(value))
    return output


def main() -> None:
    failures: list[str] = []
    manifest = ROOT / "MANIFEST.sha256"
    expected_paths: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        expected_paths.add(relative)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and not ignored_generated_file(path)
    }
    for relative in sorted(actual_paths - expected_paths):
        failures.append(f"unmanifested file: {relative}")
    for relative in sorted(expected_paths - actual_paths):
        failures.append(f"manifest entry is not a delivery file: {relative}")

    selections = json.loads((ROOT / "manifest/selections.json").read_text(encoding="utf-8"))
    for relative, expected in selected_hashes(selections):
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"selected checkpoint missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"selected checkpoint mismatch: {relative}")

    if failures:
        raise SystemExit("verification failed:\n- " + "\n- ".join(failures))
    count = len(manifest.read_text(encoding="utf-8").splitlines())
    print(f"PASS: {count} manifested files and all selected checkpoint hashes verified")


if __name__ == "__main__":
    main()
