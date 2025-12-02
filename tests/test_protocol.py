import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from supervisor import ProtocolCoordinator, SessionRecorder, build_reviewer_briefing


class StubAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    async def request(self, prompt: str) -> str:
        self.payloads.append(prompt)
        if not self.responses:
            raise AssertionError("No responses left for agent")
        return self.responses.pop(0)


class RaisingAgent:
    def __init__(self, message: str):
        self.message = message

    async def request(self, prompt: str) -> str:
        raise RuntimeError(self.message)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            (repo_dir / "foo.txt").write_text("initial content\n", encoding="utf-8")
            log_dir = repo_dir / "logs"

            reviewer = StubAgent(
                [
                    """SUMMARY: Need tweak\nPROMPT: Update foo.txt\nFILES: foo.txt\n""".strip(),
                ]
            )
            builder = StubAgent(["Turn1: Updated file per reviewer request"])

            session_dir = log_dir / "session-test"
            recorder = SessionRecorder(log_dir, session_dir=session_dir, save_git_snapshots=False)
            coordinator = ProtocolCoordinator(
                builder=builder,
                reviewer=reviewer,
                objective="Test resume",
                reviewer_marker="<<R>>",
                builder_marker="<<B>>",
                max_turns=1,
                turn_timeout=5,
                repo_path=repo_dir,
                status_lines=5,
                diff_lines=5,
                file_excerpt_lines=5,
                builder_tools=[],
                builder_tool_timeout=5,
                tool_output_lines=20,
                recorder=recorder,
                auto_commit_each_turn=False,
                auto_commit_final=False,
                auto_push_final=False,
                commit_template="Supervisor turn {turn}",
                initial_state=None,
            )
            await coordinator.run()

            state = recorder.get_state()
            self.assertIsNotNone(state)
            assert state  # type narrowing
            self.assertEqual(state.last_turn, 1)
            self.assertIsNone(coordinator.final_summary)
            self.assertEqual(coordinator.latest_reviewer_summary, "Need tweak")

            reviewer_resume = StubAgent(
                [
                    """SUMMARY: Second task\nPROMPT: Touch foo again\nFILES: foo.txt\n""".strip(),
                    "SUMMARY: Wrap-up\nPROMPT: APPROVED",
                ]
            )
            builder_resume = StubAgent(["Turn2: Followed-up on file"])
            recorder_resume = SessionRecorder(
                log_dir,
                session_dir=session_dir,
                save_git_snapshots=False,
            )
            coordinator_resume = ProtocolCoordinator(
                builder=builder_resume,
                reviewer=reviewer_resume,
                objective="Test resume",
                reviewer_marker="<<R>>",
                builder_marker="<<B>>",
                max_turns=2,
                turn_timeout=5,
                repo_path=repo_dir,
                status_lines=5,
                diff_lines=5,
                file_excerpt_lines=5,
                builder_tools=[],
                builder_tool_timeout=5,
                tool_output_lines=20,
                recorder=recorder_resume,
                auto_commit_each_turn=False,
                auto_commit_final=False,
                auto_push_final=False,
                commit_template="Supervisor turn {turn}",
                initial_state=recorder_resume.get_state(),
            )
            await coordinator_resume.run()
            resumed_state = recorder_resume.get_state()
            self.assertIsNotNone(resumed_state)
            assert resumed_state
            self.assertEqual(resumed_state.last_turn, 2)

    async def test_builder_error_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            log_dir = repo_dir / "logs"
            reviewer = StubAgent([
                "SUMMARY: Task\nPROMPT: Do a thing\nFILES: foo.txt",
            ])
            builder = RaisingAgent("failure during build")
            recorder = SessionRecorder(log_dir, save_git_snapshots=False)
            coordinator = ProtocolCoordinator(
                builder=builder,
                reviewer=reviewer,
                objective="Handle builder failures",
                reviewer_marker="<<R>>",
                builder_marker="<<B>>",
                max_turns=1,
                turn_timeout=5,
                repo_path=repo_dir,
                status_lines=5,
                diff_lines=5,
                file_excerpt_lines=5,
                builder_tools=[],
                builder_tool_timeout=5,
                tool_output_lines=20,
                recorder=recorder,
                auto_commit_each_turn=False,
                auto_commit_final=False,
                auto_push_final=False,
                commit_template="Supervisor turn {turn}",
                initial_state=None,
            )
            await coordinator.run()
            state = recorder.get_state()
            self.assertIsNotNone(state)
            assert state
            self.assertEqual(state.last_turn, 1)
            self.assertIn("Builder error", state.latest_report)
            payload = recorder.jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0]
            data = json.loads(payload)
            self.assertIn("failure during build", data["builder_report"])

    async def test_reviewer_error_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            log_dir = repo_dir / "logs"
            reviewer = RaisingAgent("reviewer crashed")
            builder = StubAgent([])
            recorder = SessionRecorder(log_dir, save_git_snapshots=False)
            coordinator = ProtocolCoordinator(
                builder=builder,
                reviewer=reviewer,
                objective="Handle reviewer failures",
                reviewer_marker="<<R>>",
                builder_marker="<<B>>",
                max_turns=1,
                turn_timeout=5,
                repo_path=repo_dir,
                status_lines=5,
                diff_lines=5,
                file_excerpt_lines=5,
                builder_tools=[],
                builder_tool_timeout=5,
                tool_output_lines=20,
                recorder=recorder,
                auto_commit_each_turn=False,
                auto_commit_final=False,
                auto_push_final=False,
                commit_template="Supervisor turn {turn}",
                initial_state=None,
            )
            await coordinator.run()
            state = recorder.get_state()
            self.assertIsNotNone(state)
            assert state
            self.assertEqual(state.last_turn, 1)
            self.assertIn("Reviewer turn failed", state.latest_reviewer_summary)

    async def test_auto_push_runs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            log_dir = repo_dir / "logs"
            reviewer = StubAgent(["SUMMARY: Good\nPROMPT: APPROVED"])
            builder = StubAgent([])
            recorder = SessionRecorder(log_dir, save_git_snapshots=False)
            with mock.patch("supervisor.auto_push_changes", return_value=True) as push:
                coordinator = ProtocolCoordinator(
                    builder=builder,
                    reviewer=reviewer,
                    objective="Push on approval",
                    reviewer_marker="<<R>>",
                    builder_marker="<<B>>",
                    max_turns=1,
                    turn_timeout=5,
                    repo_path=repo_dir,
                    status_lines=5,
                    diff_lines=5,
                    file_excerpt_lines=5,
                    builder_tools=[],
                    builder_tool_timeout=5,
                    tool_output_lines=20,
                    recorder=recorder,
                    auto_commit_each_turn=False,
                    auto_commit_final=False,
                    auto_push_final=True,
                    commit_template="Supervisor turn {turn}",
                    initial_state=None,
                )
                await coordinator.run()
            push.assert_called_once_with(repo_dir.resolve())


class ReviewerBriefingTests(unittest.TestCase):
    def test_briefing_has_no_angle_brackets(self) -> None:
        marker = "<<REVIEWER_DONE>>"
        briefing = build_reviewer_briefing("Investigate crashes", marker)
        sanitized = briefing.replace(marker, "")
        self.assertNotIn("<", sanitized)
        self.assertNotIn(">", sanitized)
        self.assertIn("Supervisor Objective: Investigate crashes", briefing)
        self.assertTrue(all(ord(ch) < 128 for ch in briefing), "Briefing must remain ASCII-only")


if __name__ == "__main__":
    unittest.main()
