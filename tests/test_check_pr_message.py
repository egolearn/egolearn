"""The PR message checker: shape rules from CONTRIBUTING.md, "The PR message"."""

# Adapted from agentic-learning-ai-lab/jepa-agent PR #72, commit a8d22e2ce0f4.
# Copyright (c) 2026 Agentic Learning AI Lab, New York University (MIT; see LICENSE).

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pr_message.py"
_spec = importlib.util.spec_from_file_location("check_pr_message", SCRIPT)
assert _spec is not None and _spec.loader is not None
check_pr_message = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_pr_message)
check = check_pr_message.check

GOOD = """## Summary

The problem and the outcome.

- `a.py`: does this now.

Part of #1.

## Test plan

- `uv run pytest` passes.

## Checklist

- [x] No secrets, credentials, or cluster-specific personal paths
- [x] No binaries or files >500 KB
- [x] `uv.lock` unchanged: no dependency change
- [x] CHANGELOG.md updated under [Unreleased]
- [x] Ported third-party-derived code keeps its attribution header
"""


def errors(title: str, body: str) -> list[str]:
    return check(title, body)[0]


def test_good_message_passes() -> None:
    assert check("Add a thing", GOOD) == ([], [])


def test_extra_section_between_summary_and_test_plan_passes() -> None:
    body = GOOD.replace("## Test plan", "## Results\n\n| a | b |\n\n## Test plan")
    assert errors("Add a thing", body) == []


def test_section_after_checklist_fails() -> None:
    assert "Checklist must be the last section" in errors("t", GOOD + "\n## Notes\n\nx\n")


def test_crlf_body_passes() -> None:
    assert errors("t", GOOD.replace("\n", "\r\n")) == []


def test_subheading_inside_summary_is_not_a_section() -> None:
    body = GOOD.replace("- `a.py`", "### Detail\n\n- `a.py`")
    assert errors("t", body) == []


def test_unticked_indented_box_fails() -> None:
    body = GOOD.replace("- [x] No binaries", "- [ ] extra\n- [x] No binaries")
    assert "1 checklist box(es) unticked" in errors("t", body)


def test_empty_body_fails_without_crashing() -> None:
    assert "missing section '## Summary'" in errors("t", "")


def test_empty_optional_section_fails() -> None:
    body = GOOD.replace("## Test plan", "## Decisions\n\n## Test plan")
    assert "section '## Decisions' is empty; fill it or drop it" in errors("t", body)


def test_title_length_boundary() -> None:
    assert errors("x" * 72, GOOD) == []
    assert errors("x" * 73, GOOD) == ["title is 73 characters, max 72"]


def test_title_rules() -> None:
    assert "title has a period" in errors("Add a thing.", GOOD)
    assert "title carries an issue number; put it in the body" in errors("Fix #12", GOOD)
    assert "title says WIP" in errors("Add a thing (wip)", GOOD)
    assert "title has a period" in errors("Fix config. Preserve defaults", GOOD)
    assert errors("Bump to v0.1 defaults", GOOD) == []
    assert "title is empty" in errors("  ", GOOD)
    assert "title must be one line" in errors("a\rb", GOOD)
    assert "title must be one line" in errors("\nAdd a thing", GOOD)


def test_duplicate_sections_fail_before_order_checks() -> None:
    body = GOOD + "\n## Checklist\n\n- [x] again\n"
    assert errors("t", body) == ["section '## Checklist' appears more than once"]


def test_checklist_topics_and_bullet_styles() -> None:
    assert "Checklist lacks a ticked box about uv.lock" in errors(
        "t", GOOD.replace("- [x] `uv.lock` unchanged: no dependency change\n", "")
    )
    assert errors("t", GOOD + "* [ ] extra\n") == ["1 checklist box(es) unticked"]
    assert errors("t", GOOD.replace("- [x] No binaries", "* [X] No binaries")) == []
    assert errors("t", GOOD.replace("- [x] No binaries", "+ [x] No binaries")) == []
    assert "1 checklist box(es) unticked" in errors("t", GOOD + "1. [ ] numbered\n")
    two_topics = "- [x] `uv.lock` unchanged and CHANGELOG.md updated\n"
    one_box = GOOD.replace("- [x] `uv.lock` unchanged: no dependency change\n", two_topics)
    one_box = one_box.replace("- [x] CHANGELOG.md updated under [Unreleased]\n", "")
    assert "Checklist needs a separate ticked box for each of the five topics" in errors(
        "t", one_box
    )


def test_checklist_notes_may_mention_other_topics() -> None:
    secrets = "- [x] No secrets, credentials, or cluster-specific personal paths"
    assert errors("t", GOOD.replace(secrets, secrets + " (also checked uv.lock)")) == []
    assert errors("t", GOOD.replace(">500 KB", ">500KB")) == []


def test_fenced_code_is_not_a_paragraph() -> None:
    body = GOOD.replace("The problem and the outcome.", "```python\nprint(42)\n```")
    body = body.replace("Part of #1.\n", "")
    assert "Summary needs a paragraph and at least one bullet" in errors("t", body)


def test_fence_closes_only_on_a_compatible_delimiter() -> None:
    nested = "````markdown\n```python\nprint(42)\n```\n````"
    body = GOOD.replace("The problem and the outcome.", nested).replace("Part of #1.\n", "")
    assert "Summary needs a paragraph and at least one bullet" in errors("t", body)
    example = "The problem and the outcome.\n\n````\n```\nunclosed inner fence\n````"
    assert check("t", GOOD.replace("The problem and the outcome.", example)) == ([], [])


def test_experiment_diff_wants_a_results_section() -> None:
    warning = check_pr_message.NO_RESULTS
    assert check("t", GOOD, ["projects/x/run.py"])[1] == [warning]
    assert check("t", GOOD, ["src/egolearn/configs/train.yaml"])[1] == [warning]
    assert check("t", GOOD, ["src/egolearn/configs/train.yml"])[1] == [warning]
    assert check("t", GOOD, ["configs/train.yaml"])[1] == [warning]
    assert check("t", GOOD, ["./projects/x/README.md"])[1] == [warning]
    assert check("t", GOOD, ["src/egolearn/pipeline.py"])[1] == []
    assert check("t", GOOD, ["docs/configs.md", "docs/projects/notes.md"])[1] == []
    with_results = GOOD.replace("## Test plan", "## Results\n\nNot run yet.\n\n## Test plan")
    assert check("t", with_results, ["projects/x/run.py"]) == ([], [])


def test_headings_inside_a_code_example_do_not_count() -> None:
    warning = check_pr_message.NO_RESULTS
    example = "````markdown\n## Results\n\nNot run yet.\n````\n\n## Test plan"
    fenced = GOOD.replace("## Test plan", example)
    assert check("t", fenced, ["projects/x/run.py"]) == ([], [warning])
    fenced_checklist = GOOD.replace("## Checklist", "```\n## Checklist\n```\n\n## Checklist")
    assert errors("t", fenced_checklist) == []


def test_a_section_of_only_fenced_code_is_not_empty() -> None:
    fenced_plan = GOOD.replace("- `uv run pytest` passes.", "```\nuv run pytest\n```")
    assert check("t", fenced_plan) == ([], [])
    for empty in ("```\n```", "```\n   \n```", "~~~\n\n~~~"):
        body = GOOD.replace("- `uv run pytest` passes.", empty)
        assert "section '## Test plan' is empty; fill it or drop it" in errors("t", body)


def test_issue_reference_alone_is_not_a_paragraph() -> None:
    message = "Summary needs a paragraph and at least one bullet"
    assert message in errors("t", GOOD.replace("The problem and the outcome.\n\n", ""))
    fenced = GOOD.replace("The problem and the outcome.", "```\ncode\n```")
    assert message in errors("t", fenced)
    assert errors("t", GOOD.replace("Part of #1.", "Part of #1 and #2, closes #3.")) == []


def test_indented_code_is_not_a_checklist() -> None:
    lines = [("    " + line if line.startswith("- [x]") else line) for line in GOOD.splitlines()]
    indented = "\n".join(lines)
    assert "Checklist lacks a ticked box about secrets" in errors("t", indented + "\n")
    assert errors("t", GOOD.replace("- [x] No binaries", "  - [x] No binaries")) == []
    nested = GOOD.replace("- [x] No binaries", "    - [ ] pending\n- [x] No binaries")
    assert "1 checklist box(es) unticked" in errors("t", nested)
    nested_ok = GOOD.replace("- [x] No binaries", "    - [x] done\n- [x] No binaries")
    assert errors("t", nested_ok) == []
    tabbed = [("\t" + line if line.startswith("- [x]") else line) for line in GOOD.splitlines()]
    assert "Checklist lacks a ticked box about secrets" in errors("t", "\n".join(tabbed) + "\n")
    fence_in_item = "    ```text\n    example\n    ```\n    - [ ] pending\n- [x] No binaries"
    with_fence = GOOD.replace("- [x] No binaries", fence_in_item)
    assert "1 checklist box(es) unticked" in errors("t", with_fence)


def test_an_indented_fence_is_code_not_a_fence() -> None:
    literal = "The problem and the outcome.\n\n    ```\n    a literal fence in a code block"
    assert check("t", GOOD.replace("The problem and the outcome.", literal)) == ([], [])


def test_a_comment_shown_inside_a_fence_is_content() -> None:
    example = "The problem and the outcome.\n\n```html\n<!-- an example comment -->\n```"
    assert check("t", GOOD.replace("The problem and the outcome.", example)) == ([], [])


def test_body_starts_with_summary() -> None:
    assert "body must start with '## Summary', nothing before it" in errors("t", "intro\n\n" + GOOD)


def test_summary_needs_a_paragraph_and_a_bullet() -> None:
    assert "Summary needs a paragraph and at least one bullet" in errors(
        "t", GOOD.replace("- `a.py`: does this now.\n", "Another sentence.\n")
    )
    heading_only = GOOD.replace("The problem and the outcome.", "### Detail")
    heading_only = heading_only.replace("Part of #1.\n", "")
    assert "Summary needs a paragraph and at least one bullet" in errors("t", heading_only)
    assert errors("t", GOOD.replace("- `a.py`", "+ `a.py`")) == []


def padded(bullets: int) -> tuple[str, int]:
    padding = "".join(f"- bullet {i}\n" for i in range(bullets))
    body = GOOD.replace("Part of #1.", padding + "Part of #1.")
    return body, sum(1 for line in body.splitlines() if line.strip())


def test_long_body_warns_then_fails() -> None:
    body, n = padded(40)
    assert 40 < n <= 60
    assert check("t", body) == ([], [f"body has {n} non-blank lines; aim under 40"])
    body, n = padded(60)
    assert n > 60
    assert errors("t", body) == [f"body has {n} non-blank lines, max 60"]


def test_title_repeated_as_a_heading_fails() -> None:
    assert "title repeated as a heading" in errors("Add a thing", "## Add a thing\n\n" + GOOD)


def test_history_section_and_template_comment_fail() -> None:
    body = GOOD.replace("## Test plan", "## Review history\n\nr1\n\n## Test plan")
    assert "section '## Review history' does not belong in a PR message" in errors("t", body)
    comment = "HTML comments left in the body (the template's, or any other)"
    assert comment in errors("t", GOOD + "<!-- x -->\n")


def test_missing_issue_reference_is_a_warning() -> None:
    warning = "no issue reference in Summary (Closes #N or Part of #N, when there is one)"
    assert check("t", GOOD.replace("Part of #1.", "Done."))[1] == [warning]
    moved = GOOD.replace("Part of #1.", "Done.").replace(
        "## Checklist", "Part of #1.\n\n## Checklist"
    )
    assert check("t", moved)[1] == [warning]


def test_attribution_box_is_required() -> None:
    body = GOOD.replace("- [x] Ported third-party-derived code keeps its attribution header\n", "")
    assert errors("t", body) == ["Checklist lacks a ticked box about attribution"]


@pytest.mark.parametrize("section", ["Summary", "Test plan", "Checklist"])
def test_required_sections_cannot_be_omitted(section: str) -> None:
    assert f"missing section '## {section}'" in errors("t", GOOD.replace(f"## {section}", ""))


def test_sections_cannot_interrupt_test_plan_and_checklist() -> None:
    body = GOOD.replace("## Checklist", "## Notes\n\nMore context.\n\n## Checklist")
    assert "nothing goes between Test plan and Checklist" in errors("t", body)


def test_leading_blanks_do_not_hide_repeated_title() -> None:
    assert errors("t", "\n\n" + GOOD) == []
    assert "body starts with a top-level heading (the title repeated)" in errors(
        "t", "\n# t\n\n" + GOOD
    )


def run_draft(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "pr body.md"
    path.write_text(body, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "check PR message format",
            "--body-file",
            str(path),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_local_cli_accepts_an_empty_file_list(tmp_path: Path) -> None:
    result = run_draft(tmp_path, GOOD, "--files")
    assert result.returncode == 0, result.stderr
    assert "PR message: ok" in result.stdout
    assert "warning:" not in result.stdout


def test_cli_warnings_succeed_but_errors_fail(tmp_path: Path) -> None:
    result = run_draft(tmp_path, GOOD, "--files", "projects/an experiment/run.yaml")
    assert result.returncode == 0, result.stderr
    assert f"warning: {check_pr_message.NO_RESULTS}" in result.stdout
    result = run_draft(tmp_path, GOOD.replace("- [x]", "- [ ]", 1))
    assert result.returncode == 1
    assert "error: 1 checklist box(es) unticked" in result.stdout
    assert "PR message: FAIL" in result.stdout


def test_unfilled_template_is_rejected(tmp_path: Path) -> None:
    template = Path(__file__).resolve().parents[1] / ".github" / "pull_request_template.md"
    result = run_draft(tmp_path, template.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert "HTML comments left in the body" in result.stdout
    assert "5 checklist box(es) unticked" in result.stdout


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "give --pr N, or --title and --body-file"),
        (["--pr", "0"], "--pr must be a positive number"),
        (["--pr", "1", "--title", "t"], "--pr cannot be combined"),
        (["--pr", "1", "--files"], "--pr cannot be combined"),
        (["--title", "t", "--body-file", "missing.md"], "could not read PR message"),
    ],
)
def test_cli_reports_usage_and_file_errors(tmp_path: Path, args: list[str], message: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_existing_pr_reads_changed_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read_pr(command: list[str], **kwargs: object) -> str:
        assert command == ["gh", "pr", "view", "26", "--json", "title,body,files"]
        return json.dumps(
            {"title": "check PR messages", "body": GOOD, "files": [{"path": "configs/a.yaml"}]}
        )

    monkeypatch.setattr(sys, "argv", ["check_pr_message.py", "--pr", "26"])
    monkeypatch.setattr(check_pr_message.subprocess, "check_output", read_pr)
    assert check_pr_message.main() == 0
    assert f"warning: {check_pr_message.NO_RESULTS}" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["missing", "auth", "json"])
def test_existing_pr_reports_read_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    def read_pr(command: list[str], **kwargs: object) -> str:
        if failure == "missing":
            raise FileNotFoundError("gh is not installed")
        if failure == "auth":
            raise subprocess.CalledProcessError(1, command, stderr="gh authentication required")
        return "invalid json"

    monkeypatch.setattr(sys, "argv", ["check_pr_message.py", "--pr", "26"])
    monkeypatch.setattr(check_pr_message.subprocess, "check_output", read_pr)
    with pytest.raises(SystemExit) as exc:
        check_pr_message.main()
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
