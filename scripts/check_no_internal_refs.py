#!/usr/bin/env python3
"""Guard: keep internal/development references out of the public tree.

Fails (exit 1) if any tracked file reintroduces:
  - local-only doc paths that don't ship publicly (docs/_internal lives behind
    the old names ``docs/superpowers/`` and ``docs/claude/``),
  - private context-file names (``CLAUDE.md`` / ``AGENTS.md``),
  - internal tooling / process / persona terms, or
  - wiki-style ``[[concept]]`` links in Markdown.

Public design docs live under ``docs/research/``; day-to-day development notes
and design specs/plans stay local under ``docs/_internal/`` (gitignored). This
check stops those internal artifacts from leaking back into committed files.

Run from the repository root (CI invokes it right after checkout).
"""

from __future__ import annotations

import re
import subprocess
import sys

# (human-readable label, compiled pattern, optional path suffix the rule is limited to)
RULES: list[tuple[str, re.Pattern[str], str | None]] = [
    ("local-only path 'docs/superpowers/'", re.compile(r"docs/superpowers/"), None),
    ("renamed path 'docs/claude/' (now docs/research/)", re.compile(r"docs/claude/"), None),
    ("internal tooling name 'superpowers'", re.compile(r"superpowers"), None),
    ("private context file CLAUDE.md / AGENTS.md", re.compile(r"\b(?:CLAUDE|AGENTS)\.md\b"), None),
    ("internal process term 'subagent'", re.compile(r"\bsub-?agent\b", re.IGNORECASE), None),
    ("internal persona 'Hypatia'", re.compile(r"\bHypatia\b"), None),
    (
        "AI-tool attribution",
        re.compile(r"Claude\s+(?:Code|Design)|Co-Authored-By:\s*Claude|Generated with .{0,8}Claude"),
        None,
    ),
    ("wiki-style [[concept]] link", re.compile(r"\[\[[a-z][a-z0-9-]{4,}\]\]"), ".md"),
]

# Files that legitimately contain these tokens and must be skipped:
#   - this guard (it *defines* the patterns),
#   - .gitignore (the CLAUDE.md / AGENTS.md ignore patterns are required),
#   - the CI workflow (it invokes this guard by name).
ALLOWLIST_EXACT = {"scripts/check_no_internal_refs.py", ".gitignore"}
ALLOWLIST_PREFIXES = (".github/workflows/",)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def is_allowlisted(path: str) -> bool:
    return path in ALLOWLIST_EXACT or path.startswith(ALLOWLIST_PREFIXES)


def main() -> int:
    violations: list[str] = []
    for path in tracked_files():
        if is_allowlisted(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or unreadable — nothing to lint
        for label, pattern, suffix in RULES:
            if suffix is not None and not path.endswith(suffix):
                continue
            for lineno, line in enumerate(lines, start=1):
                if pattern.search(line):
                    violations.append(f"{path}:{lineno}: {label}")

    if violations:
        print("Internal/development references found in tracked files:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nMove design specs/plans to docs/_internal/ (gitignored), put public "
            "notes under docs/research/, and use neutral wording. See "
            "scripts/check_no_internal_refs.py for the rules.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no internal/development references in {len(tracked_files())} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
