"""Check a pull request message against CONTRIBUTING.md, "The PR message".

    # Bash 4.4+: preserve filenames containing whitespace or glob characters.
    mapfile -d '' -t changed_files < <(git diff --name-only -z origin/main...HEAD)
    python scripts/check_pr_message.py --title "describe the change" --body-file body.md \\
        --files "${changed_files[@]}"
    python scripts/check_pr_message.py --pr 67

Checks the shape only: title, the sections and their order, ticked checklist
boxes, leftover comments, length, and a Results section when the changed files
touch projects/ or a configs/*.yaml or *.yml file (the file list comes from --files or
from gh). Fenced code is content, never structure. Exit 1 on any error;
warnings do not fail. It cannot judge the prose.

Markdown is parsed approximately: a fence is indented at most three spaces and
closes on a matching delimiter; four spaces open a code block unless the line
continues a list; tabs count as four spaces; code indented inside a list item
is read as part of the list.
"""

# Adapted from agentic-learning-ai-lab/jepa-agent PR #72, commit a8d22e2ce0f4.
# Copyright (c) 2026 Agentic Learning AI Lab, New York University (MIT; see LICENSE).

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys

REQUIRED = ["Summary", "Test plan", "Checklist"]
FORBIDDEN = re.compile(r"review history|changelog", re.I)
ISSUE_REF = re.compile(r"\b(Closes|Part of)\s+#\d+")
MAX_TITLE = 72
TARGET_LINES = 40
MAX_LINES = 60
LIST = r"(?:[-*+]|\d+[.)])"
BOX = re.compile(rf"^\s*{LIST}\s+\[([ xX])\]\s*(.*)$")
ITEM = re.compile(rf"^\s*{LIST}\s")
BULLET = re.compile(rf"^ {{0,3}}{LIST}\s")
NOT_PROSE = re.compile(r"^\s*(#|\||\x00)|^ {4,}")
ISSUE_ONLY = re.compile(r"^\s*(?:(?:Closes|Part of)\s+#\d+|and|[,.\s])+$")
EXPERIMENT_PATH = re.compile(r"^projects/|(^|/)configs/.*\.ya?ml$", re.DOTALL)
NO_RESULTS = "the diff touches projects/ or YAML configs: add a Results section, or say why not"
CHECKLIST_TOPICS = {
    "secrets": r"secrets",
    "500 KB": r"500\s*kb",
    "uv.lock": r"uv\.lock",
    "CHANGELOG": r"changelog",
    "attribution": r"attribution",
}


def sections(lines: list[str]) -> list[tuple[int, str]]:
    heads = []
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            heads.append((i, m.group(1)))
    return heads


FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")  # deeper than three spaces is code, not a fence


MASK = "\x00"


def mask_fences(lines: list[str]) -> list[str]:
    """Mask fenced blocks in place: their text is content, never structure.

    Delimiter lines and blank lines inside a fence become blank, so an empty
    fence is empty; other fenced lines become a marker. A fence closes only on
    the same character, at least as long, alone on its line.
    """
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(1)[0], len(m.group(1)))
                out.append("")
            else:
                out.append(line)
        elif m is not None and not m.group(2).strip() and m.group(1)[0] == fence[0]:
            if len(m.group(1)) >= fence[1]:
                fence = None
                out.append("")
            else:
                out.append(" " * indent + MASK)
        else:
            out.append(" " * indent + MASK if line.strip() else "")
    return out


def boxes_in(lines: list[str]) -> list[re.Match[str]]:
    """Task boxes in a section: at up to three spaces, or nested under a list item.

    Four or more spaces of indentation open an indented code block, unless the
    line continues a list.
    """
    out: list[re.Match[str]] = []
    in_list = False
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        item = ITEM.match(line) is not None
        if item and (indent <= 3 or in_list):
            if m := BOX.match(line):
                out.append(m)
            in_list = True
        else:
            in_list = in_list and indent > 0
    return out


def body_of(lines: list[str], heads: list[tuple[int, str]], name: str) -> list[str]:
    for k, (start, head) in enumerate(heads):
        if head == name:
            end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
            return [line for line in lines[start + 1 : end] if line.strip()]
    return []


def check(title: str, body: str, files: list[str] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not title.strip():
        errors.append("title is empty")
    if re.search(r"[\r\n]", title):
        errors.append("title must be one line")
    if len(title) > MAX_TITLE:
        errors.append(f"title is {len(title)} characters, max {MAX_TITLE}")
    if re.search(r"\.(\s|$)", title):
        errors.append("title has a period")
    if re.search(r"#\d+", title):
        errors.append("title carries an issue number; put it in the body")
    if re.search(r"\bwip\b", title, re.I):
        errors.append("title says WIP")

    raw = [line.expandtabs(4) for line in body.splitlines()]
    nonblank = [line for line in raw if line.strip()]
    lines = mask_fences(raw)  # headings and boxes inside a code example do not count
    if len(nonblank) > MAX_LINES:
        errors.append(f"body has {len(nonblank)} non-blank lines, max {MAX_LINES}")
    elif len(nonblank) > TARGET_LINES:
        warnings.append(f"body has {len(nonblank)} non-blank lines; aim under {TARGET_LINES}")
    if nonblank and nonblank[0].startswith("# "):
        errors.append("body starts with a top-level heading (the title repeated)")
    elif nonblank and nonblank[0] != "## Summary":
        errors.append("body must start with '## Summary', nothing before it")
    if any("<!--" in line for line in lines):  # a comment shown inside a fence is content
        errors.append("HTML comments left in the body (the template's, or any other)")

    heads = sections(lines)
    names = [head for _, head in heads]
    if any(head.strip().lower() == title.strip().lower() for head in names):
        errors.append("title repeated as a heading")
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"section '## {name}' appears more than once")
    if len(set(names)) != len(names):
        return errors, warnings
    for required in REQUIRED:
        if required not in names:
            errors.append(f"missing section '## {required}'")
    if all(required in names for required in REQUIRED):
        if names[0] != "Summary":
            errors.append("Summary must be the first section")
        if names[-1] != "Checklist":
            errors.append("Checklist must be the last section")
        if names.index("Checklist") != names.index("Test plan") + 1:
            errors.append("nothing goes between Test plan and Checklist")
    for head in names:
        if FORBIDDEN.search(head):
            errors.append(f"section '## {head}' does not belong in a PR message")
    for head in names:
        if not body_of(lines, heads, head):
            errors.append(f"section '## {head}' is empty; fill it or drop it")
    if "Summary" in names:
        summary = body_of(lines, heads, "Summary")
        bullets = [line for line in summary if BULLET.match(line)]
        prose = [
            line
            for line in summary
            if not BULLET.match(line) and not NOT_PROSE.match(line) and not ISSUE_ONLY.match(line)
        ]
        if not bullets or not prose:
            errors.append("Summary needs a paragraph and at least one bullet")
        if not ISSUE_REF.search("\n".join(summary)):
            warnings.append(
                "no issue reference in Summary (Closes #N or Part of #N, when there is one)"
            )
    if "Results" not in names and any(
        EXPERIMENT_PATH.search(f.removeprefix("./")) for f in files or []
    ):
        warnings.append(NO_RESULTS)
    if "Checklist" in names:
        boxes = boxes_in(body_of(lines, heads, "Checklist"))
        unticked = [m for m in boxes if m.group(1) == " "]
        if unticked:
            errors.append(f"{len(unticked)} checklist box(es) unticked")
        ticked = [m.group(2) for m in boxes if m.group(1) != " "]
        candidates = {
            topic: [i for i, text in enumerate(ticked) if re.search(pattern, text, re.I)]
            for topic, pattern in CHECKLIST_TOPICS.items()
        }
        for topic, found in candidates.items():
            if not found:
                errors.append(f"Checklist lacks a ticked box about {topic}")
        if all(candidates.values()) and not any(
            len(set(pick)) == len(pick) for pick in itertools.product(*candidates.values())
        ):
            errors.append("Checklist needs a separate ticked box for each of the five topics")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, help="check an existing pull request through gh")
    parser.add_argument("--title")
    parser.add_argument("--body-file")
    parser.add_argument("--files", nargs="*", help="changed paths, for the Results check")
    args = parser.parse_args()
    if args.pr is not None and (
        args.title is not None or args.body_file is not None or args.files is not None
    ):
        parser.error("--pr cannot be combined with --title, --body-file, or --files")
    try:
        if args.pr is not None:
            if args.pr < 1:
                parser.error("--pr must be a positive number")
            out = subprocess.check_output(
                ["gh", "pr", "view", str(args.pr), "--json", "title,body,files"],
                text=True,
                stderr=subprocess.PIPE,
            )
            data = json.loads(out)
            title, body = data["title"], data["body"] or ""
            files = [f["path"] for f in data["files"]]
        elif args.title is not None and args.body_file:
            with open(args.body_file, encoding="utf-8") as f:
                title, body = args.title, f.read()
            files = args.files
        else:
            parser.error("give --pr N, or --title and --body-file")
    except subprocess.CalledProcessError as exc:
        parser.error(f"gh failed: {exc.stderr.strip()}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"could not read PR message: {exc}")
    errors, warnings = check(title, body, files)
    for error in errors:
        print(f"error: {error}")
    for warning in warnings:
        print(f"warning: {warning}")
    verdict = "FAIL" if errors else "ok"
    nonblank = sum(1 for line in body.splitlines() if line.strip())
    print(f"PR message: {verdict} ({nonblank} non-blank lines, title {len(title)} chars)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
