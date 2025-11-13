import tempfile
from pathlib import Path
import unittest

from supervisor import ProtocolCoordinator, SessionRecorder


class StubAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    async def send(self, text: str) -> None:  # pragma: no cover - trivial passthrough
        self.sent.append(text)

    async def read_until_marker(self, marker: str, timeout=None) -> str:
        if not self.responses:
            raise AssertionError("No responses left for agent")
        return self.responses.pop(0)


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
                commit_template="Supervisor turn {turn}",
                initial_state=recorder_resume.get_state(),
            )
            await coordinator_resume.run()
            resumed_state = recorder_resume.get_state()
            self.assertIsNotNone(resumed_state)
            assert resumed_state
            self.assertEqual(resumed_state.last_turn, 2)


if __name__ == "__main__":
    unittest.main()
