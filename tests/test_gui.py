import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gui


class DummyVar:
    def __init__(self, value=None):
        self._value = value
        self._traces = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for callback in self._traces:
            callback(None, None, None)

    def trace_add(self, mode, callback):
        self._traces.append(callback)


class DummyBooleanVar(DummyVar):
    pass


class DummyText:
    def __init__(self):
        self.text = ""
        self.state = None

    def configure(self, **kwargs):
        state = kwargs.get("state")
        if state is not None:
            self.state = state

    def insert(self, index, value):
        self.text += value

    def count(self, start, end, mode):
        return (len(self.text),)

    def delete(self, start, end=None):
        if start == "1.0" and end and end.startswith("1.0+"):
            number = int(end.split("+")[1][:-1])
            self.text = self.text[number:]
        elif start.startswith("1.0+"):
            number = int(start.split("+")[1][:-1])
            self.text = self.text[number:]
        elif end and end.startswith("1.0+"):
            number = int(end.split("+")[1][:-1])
            self.text = self.text[number:]
        else:
            self.text = ""

    def see(self, index):
        pass

    def get(self, start, end):
        return self.text


class DummyButton:
    def __init__(self):
        self.state = None
        self.text = None

    def config(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]


def make_tab() -> gui.SupervisorTab:
    tab = gui.SupervisorTab.__new__(gui.SupervisorTab)
    tab.output_text = DummyText()
    tab._handle_log_line = lambda line: None
    tab._context_stats = {"Builder": None, "Reviewer": None}
    tab.context_threshold_var = DummyVar("")
    tab._auto_restart_requested = False
    tab._auto_restart_pending = False
    tab._restart_reason = ""
    tab._user_stop_requested = False
    tab._stop_after_prompt_requested = False
    tab._destroyed = False
    tab.after = lambda delay, func=None, *args: func(*args) if func else None
    tab.after_cancel = lambda job: None
    tab.last_command = None
    tab.process = None
    tab.controller = mock.Mock()
    tab.prevent_screen_sleep_var = DummyBooleanVar(False)
    tab.prevent_computer_sleep_var = DummyBooleanVar(False)
    tab.auto_push_var = DummyBooleanVar(False)
    tab.carousel_var = DummyBooleanVar(False)
    tab.carousel_rotations_var = DummyVar(0)
    tab.auto_push_button = DummyButton()
    tab.start_button = DummyButton()
    tab.stop_button = DummyButton()
    tab.timer_var = DummyVar("00:00:00")
    tab._timer_job = None
    tab._timer_start = None
    tab._sleep_process = None
    tab._sleep_warning_shown = False
    tab.stop_after_prompt_button = DummyButton()
    return tab


class SupervisorTabUnitTests(unittest.TestCase):
    def test_log_line_truncates(self) -> None:
        tab = make_tab()
        long_text = "x" * (gui.OUTPUT_MAX_CHARS + 10)
        tab._log_line(long_text)
        content = tab.output_text.get("1.0", "end-1c")
        self.assertLessEqual(len(content), gui.OUTPUT_MAX_CHARS)

    def test_resolve_codex_cli(self) -> None:
        tab = make_tab()
        with self.assertRaises(FileNotFoundError):
            tab._resolve_codex_cli("/no/such/codex")
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy = Path(tmpdir) / "codex.bin"
            dummy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            dummy.chmod(0o755)
            resolved = tab._resolve_codex_cli(str(dummy))
            self.assertEqual(resolved, str(dummy))

    def test_update_context_triggers_restart_after_turn(self) -> None:
        tab = make_tab()
        tab.context_threshold_var = DummyVar("50")
        tab.process = object()
        tab.last_command = ["python", "gui.py"]
        with mock.patch.object(tab, "stop_supervisor") as stop:
            line = "[Builder] Turn completed (output tokens: 10) | context left: 100 tokens (~40.0%)"
            tab._update_context_from_line(line)
            stop.assert_not_called()
            self.assertTrue(tab._auto_restart_requested)
            wait_line = "[Supervisor] Turn 5: waiting for reviewer instructions..."
            tab._maybe_stop_after_current_turn(wait_line)
            stop.assert_called_once_with(auto=True)
            self.assertTrue(tab._auto_restart_pending)

    def test_update_context_ignored_without_threshold(self) -> None:
        tab = make_tab()
        tab.context_threshold_var = DummyVar("")
        tab.process = object()
        tab.last_command = ["python", "gui.py"]
        with mock.patch.object(tab, "stop_supervisor") as stop:
            tab._update_context_from_line(
                "[Reviewer] Turn completed | context left: 100 tokens (~40.0%)"
            )
            stop.assert_not_called()

    def test_stop_after_prompt_waits_for_turn_boundary(self) -> None:
        tab = make_tab()
        tab.process = object()
        with mock.patch.object(gui.messagebox, "showinfo"):
            tab.stop_after_prompt()
        self.assertTrue(tab._stop_after_prompt_requested)
        with mock.patch.object(tab, "stop_supervisor") as stop:
            tab._maybe_stop_after_current_turn("[Supervisor] Turn 3: waiting for reviewer instructions...")
            stop.assert_called_once_with()
        self.assertFalse(tab._stop_after_prompt_requested)

    def test_sleep_prevention_mac_uses_caffeinate(self) -> None:
        tab = make_tab()
        tab.process = object()
        tab.prevent_screen_sleep_var = DummyBooleanVar(True)
        tab.prevent_computer_sleep_var = DummyBooleanVar(True)
        fake_proc = mock.Mock()
        fake_proc.wait.return_value = None
        original_platform = gui.sys.platform
        with mock.patch.object(gui.subprocess, "Popen", return_value=fake_proc) as popen:
            gui.sys.platform = "darwin"
            try:
                tab._start_sleep_prevention()
            finally:
                gui.sys.platform = original_platform
        popen.assert_called_once_with(["caffeinate", "-d", "-i"])
        self.assertIs(tab._sleep_process, fake_proc)
        tab._stop_sleep_prevention()
        fake_proc.terminate.assert_called_once()

    def test_sleep_prevention_noop_off_macos(self) -> None:
        tab = make_tab()
        tab.process = object()
        tab.prevent_screen_sleep_var = DummyBooleanVar(True)
        original_platform = gui.sys.platform
        with mock.patch.object(gui.subprocess, "Popen") as popen:
            gui.sys.platform = "linux"
            try:
                tab._start_sleep_prevention()
            finally:
                gui.sys.platform = original_platform
        popen.assert_not_called()
        self.assertIsNone(tab._sleep_process)

    def test_carousel_restarts_after_clean_exit(self) -> None:
        tab = make_tab()
        tab.carousel_var = DummyBooleanVar(True)
        tab.last_command = ["python", "gui.py"]
        restarts: list[str] = []
        tab._restart_supervisor = lambda: restarts.append("restart")

        proc = mock.Mock()
        proc.returncode = 0
        proc.poll.return_value = 0
        tab.process = proc

        tab._check_process()

        self.assertEqual(restarts, ["restart"])
        self.assertIsNone(tab.process)
        self.assertEqual(tab.carousel_rotations_var.get(), 1)

    def test_carousel_skipped_after_manual_stop(self) -> None:
        tab = make_tab()
        tab.carousel_var = DummyBooleanVar(True)
        tab.last_command = ["python", "gui.py"]
        restarts: list[str] = []
        tab._restart_supervisor = lambda: restarts.append("restart")

        proc = mock.Mock()
        proc.returncode = 0
        proc.poll.return_value = 0
        tab.process = proc
        tab._user_stop_requested = True

        tab._check_process()

        self.assertEqual(restarts, [])
        tab.controller.tab_finished.assert_called_once_with(tab)
        self.assertEqual(tab.carousel_rotations_var.get(), 0)

    def test_carousel_rotations_not_incremented_when_disabled(self) -> None:
        tab = make_tab()
        tab.carousel_var = DummyBooleanVar(False)
        tab.last_command = ["python", "gui.py"]
        tab.carousel_rotations_var = DummyVar(0)
        restarts: list[str] = []
        tab._restart_supervisor = lambda: restarts.append("restart")

        proc = mock.Mock()
        proc.returncode = 0
        proc.poll.return_value = 0
        tab.process = proc

        tab._check_process()

        self.assertEqual(restarts, [])
        self.assertEqual(tab.carousel_rotations_var.get(), 0)

    def test_carousel_rotations_reset_helper(self) -> None:
        tab = make_tab()
        tab.carousel_rotations_var = DummyVar(5)
        tab._reset_carousel_rotations()
        self.assertEqual(tab.carousel_rotations_var.get(), 0)

    def test_carousel_rotations_increment_recovers_from_invalid_state(self) -> None:
        tab = make_tab()
        tab.carousel_rotations_var = DummyVar("not-an-int")
        tab._increment_carousel_rotations()
        self.assertEqual(tab.carousel_rotations_var.get(), 1)


class SupervisorGUIUnitTests(unittest.TestCase):
    def test_open_logs_invokes_platform_handler(self) -> None:
        opened = []
        obj = gui.SupervisorGUI.__new__(gui.SupervisorGUI)
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"

            def fake_run(args, check=False):
                opened.append(args)

            with mock.patch.object(gui, "LOGS_DIR", logs_dir), mock.patch.object(
                gui.subprocess,
                "run",
                side_effect=fake_run,
            ):
                original_platform = gui.sys.platform
                gui.sys.platform = "darwin"
                try:
                    obj.open_logs_folder()
                finally:
                    gui.sys.platform = original_platform
            self.assertTrue(logs_dir.exists())
            self.assertIn(["open", str(logs_dir)], opened)


if __name__ == "__main__":
    unittest.main()
