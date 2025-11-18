import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import sys
from unittest import mock

from supervisor import (
    EXCERPT_MAX_BYTES,
    SessionRecorder,
    ReviewerTurn,
    auto_commit_changes,
    format_commit_message,
    load_file_excerpt,
    parse_codex_exec_output,
    summarize_codex_event,
    parse_reviewer_response,
    run_tool_command,
    truncate_lines,
)


class TruncateAndParseTests(unittest.TestCase):
    def test_truncate_lines_adds_suffix_when_exceeding_limit(self) -> None:
        text = "one\ntwo\nthree\nfour"
        self.assertEqual(
            truncate_lines(text, 2),
            "one\ntwo\n... (2 more lines truncated)",
        )

    def test_parse_reviewer_response_extracts_sections_and_files(self) -> None:
        message = (
            "SUMMARY: Fix issue\n"
            "PROMPT: Change foo\n"
            "FILES: foo.py, bar.py baz.txt\n"
            "CONTEXT: Keep logging\n Extra note line\n"
            "Trailing text without header"
        )
        turn = parse_reviewer_response(message)
        self.assertEqual(turn.summary, "Fix issue")
        self.assertEqual(turn.prompt, "Change foo")
        self.assertEqual(turn.files, ["foo.py", "bar.py", "baz.txt"])
        self.assertIn("Keep logging", turn.context)
        self.assertIn("Extra note line", turn.context)

    def test_parse_codex_exec_output_extracts_agent_message(self) -> None:
        sample = "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"Result"}}',
                '{"type":"turn.completed"}',
            ]
        )
        self.assertEqual(parse_codex_exec_output(sample), "Result")

    def test_parse_codex_exec_output_falls_back_to_raw(self) -> None:
        raw = "non-json output"
        self.assertEqual(parse_codex_exec_output(raw), raw)

    def test_parse_codex_exec_output_warns_on_invalid_json(self) -> None:
        warnings = []
        parse_codex_exec_output("not-json", warn=warnings.append)
        self.assertTrue(warnings)
        self.assertIn("Failed to parse", warnings[0])

    def test_summarize_codex_event_formats_reasoning(self) -> None:
        line = '{"type":"item.completed","item":{"type":"reasoning","text":"Analyzing files"}}'
        self.assertEqual(summarize_codex_event(line), "Reasoning: Analyzing files")

    def test_summarize_codex_event_handles_command(self) -> None:
        line = (
            '{"type":"item.completed","item":{"type":"command_execution","command":"bash -lc pwd",'
            '"status":"completed","exit_code":0,"aggregated_output":"/tmp\\n"}}'
        )
        summary = summarize_codex_event(line)
        self.assertIn("Command: bash -lc pwd", summary)
        self.assertIn("exit 0", summary)
        self.assertIn("/tmp", summary)

    def test_summarize_codex_event_warns_on_invalid_json(self) -> None:
        warnings = []
        result = summarize_codex_event("not-json", warn=warnings.append)
        self.assertIsNone(result)
        self.assertTrue(warnings)

    def test_summarize_codex_event_reports_context_left(self) -> None:
        line = (
            '{"type":"turn.completed","usage":{"output_tokens":200,"total_tokens":4000,"context_window":16000}}'
        )
        summary = summarize_codex_event(line)
        self.assertIn("Turn completed (output tokens: 200)", summary)
        self.assertIn("context left", summary)


class FileExcerptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmpdir.name)
        (self.repo / "file.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_file_excerpt_truncates_and_handles_missing(self) -> None:
        name, excerpt = load_file_excerpt(self.repo, "file.txt", max_lines=2)
        self.assertEqual(name, "file.txt")
        self.assertIn("line1", excerpt)
        self.assertIn("... (1 more lines truncated)", excerpt)

        name_missing, message = load_file_excerpt(self.repo, "missing.txt", max_lines=2)
        self.assertEqual(name_missing, "missing.txt")
        self.assertEqual(message, "(File not found.)")

    def test_load_file_excerpt_rejects_paths_outside_repo(self) -> None:
        parent = self.repo.parent
        outside = os.path.relpath(parent / "elsewhere.txt", self.repo)
        name, message = load_file_excerpt(self.repo, outside, max_lines=2)
        self.assertEqual(name, outside)
        self.assertEqual(message, "(Path escapes repository root.)")

    def test_load_file_excerpt_limits_large_files(self) -> None:
        big = self.repo / "big.txt"
        big.write_text("x" * (EXCERPT_MAX_BYTES + 1), encoding="utf-8")
        name, message = load_file_excerpt(self.repo, "big.txt", max_lines=2)
        self.assertEqual(name, "big.txt")
        self.assertIn("exceeds excerpt size", message)

    def test_load_file_excerpt_detects_binary_files(self) -> None:
        binary = self.repo / "binary.bin"
        binary.write_bytes(b"\x00\x01\x02")
        name, message = load_file_excerpt(self.repo, "binary.bin", max_lines=2)
        self.assertEqual(name, "binary.bin")
        self.assertEqual(message, "(Binary file omitted.)")


class ToolCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_run_tool_command_captures_stdout_and_stderr(self) -> None:
        command = f"\"{sys.executable}\" -c \"import sys; print('out'); sys.stderr.write('err\\\\n')\""
        result = run_tool_command(
            command,
            repo_path=self.repo,
            timeout=5,
            output_lines=10,
        )
        self.assertEqual(Path(result.name.strip('"')).name, Path(sys.executable).name)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("out", result.output)
        self.assertIn("err", result.output)

    def test_run_tool_command_reports_timeout(self) -> None:
        command = f"\"{sys.executable}\" -c \"import time; time.sleep(2)\""
        start = time.monotonic()
        result = run_tool_command(
            command,
            repo_path=self.repo,
            timeout=1,
            output_lines=5,
        )
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 1)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("Command timed out", result.output)


class SessionRecorderResilienceTests(unittest.TestCase):
    def test_record_turn_handles_io_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            recorder = SessionRecorder(log_dir, save_git_snapshots=False)
            recorder.jsonl.close()
            recorder.md.close()
            recorder.jsonl = mock.Mock()
            recorder.jsonl.write.side_effect = OSError("disk full")
            recorder.jsonl.flush = mock.Mock()
            recorder.md = mock.Mock()
            recorder.md.write = mock.Mock()
            recorder.md.flush = mock.Mock()
            turn = ReviewerTurn(summary="s", prompt="p", files=[], context="")
            recorder.record_turn(
                turn=1,
                repo_context="ctx",
                reviewer_turn=turn,
                builder_report="report",
                tool_results=[],
                tool_summary="tools",
                git_snapshot=None,
            )
            state = recorder.get_state()
            self.assertIsNotNone(state)
            assert state
            self.assertEqual(state.last_turn, 1)
            self.assertEqual(state.latest_report, "report")

    def test_finalize_handles_io_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            recorder = SessionRecorder(log_dir, save_git_snapshots=False)
            recorder.jsonl.close()
            recorder.md.close()
            recorder.jsonl = mock.Mock()
            recorder.jsonl.close = mock.Mock()
            recorder.md = mock.Mock()
            recorder.md.write.side_effect = OSError("disk full")
            recorder.md.flush = mock.Mock()
            recorder.md.close = mock.Mock()
            recorder.finalize("done")


class AutoCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmpdir.name)
        subprocess.run(["git", "init"], cwd=self.repo, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo, check=True)
        (self.repo / "tracked.txt").write_text("original\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.repo, check=True, stdout=subprocess.DEVNULL)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_auto_commit_changes_creates_commit_when_dirty(self) -> None:
        (self.repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
        committed = auto_commit_changes(self.repo, "automated commit")
        self.assertTrue(committed)
        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(log, "automated commit")
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(status.strip(), "")

    def test_auto_commit_changes_noop_when_clean(self) -> None:
        committed = auto_commit_changes(self.repo, "should not commit")
        self.assertFalse(committed)


class CommitMessageTests(unittest.TestCase):
    def test_format_commit_message_replaces_placeholders(self) -> None:
        message = format_commit_message(
            "Turn {turn}: {summary} / {final_summary}",
            turn=3,
            summary="sum",
            final_summary="final",
        )
        self.assertEqual(message, "Turn 3: sum / final")

        final_message = format_commit_message(
            "Final ({turn}): {final_summary}",
            turn=None,
            summary="ignored",
            final_summary="all done",
        )
        self.assertEqual(final_message, "Final (final): all done")


if __name__ == "__main__":
    unittest.main()
