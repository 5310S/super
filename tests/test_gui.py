import sys
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


class SupervisorGUIUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.settings_path = Path(self.tmpdir.name) / "settings.json"
        self.logs_dir = Path(self.tmpdir.name) / "logs"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.obj = gui.SupervisorGUI.__new__(gui.SupervisorGUI)
        self.obj._settings_vars = {}
        self.obj.settings_path = self.settings_path
        self.obj._save_pending = False
        self.obj._loading_settings = False
        self.obj.output_text = DummyText()
        self.obj._auto_restart_requested = False
        self.obj.context_threshold_var = DummyVar("")
        self.obj._context_stats = {"Builder": None, "Reviewer": None}
        self.obj._auto_restart_pending = False
        self.obj._restart_reason = ""
        self.obj.last_command = None
        self.obj.process = None
        self.obj.prevent_screen_sleep_var = DummyBooleanVar(False)
        self.obj.prevent_computer_sleep_var = DummyBooleanVar(False)
        self.obj._sleep_process = None
        self.obj._sleep_warning_shown = False

    def test_save_and_load_settings_round_trip(self) -> None:
        vars_map = {
            "objective": DummyVar("Ship"),
            "auto": DummyBooleanVar(True),
        }
        self.obj._settings_vars = vars_map
        with mock.patch.object(gui.tk, "BooleanVar", DummyBooleanVar):
            self.obj._save_settings()
            vars_map["objective"].set("Reset")
            vars_map["auto"].set(False)
            self.obj._load_settings()
        self.assertEqual(vars_map["objective"].get(), "Ship")
        self.assertTrue(vars_map["auto"].get())

    def test_log_line_truncates(self) -> None:
        long_text = "x" * (gui.OUTPUT_MAX_CHARS + 10)
        self.obj._log_line(long_text)
        content = self.obj.output_text.get("1.0", "end-1c")
        self.assertLessEqual(len(content), gui.OUTPUT_MAX_CHARS)

    def test_resolve_codex_cli(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.obj._resolve_codex_cli("/no/such/codex")
        dummy = Path(self.tmpdir.name) / "codex.bin"
        dummy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        dummy.chmod(0o755)
        resolved = self.obj._resolve_codex_cli(str(dummy))
        self.assertEqual(resolved, str(dummy))

    def test_open_logs_invokes_platform_handler(self) -> None:
        opened = []

        def fake_run(args, check=False):
            opened.append(args)

        with mock.patch.object(gui, "LOGS_DIR", self.logs_dir), mock.patch.object(
            gui.subprocess, "run", side_effect=fake_run
        ):
            original_platform = gui.sys.platform
            gui.sys.platform = "darwin"
            try:
                self.obj._open_logs()
            finally:
                gui.sys.platform = original_platform
        self.assertTrue(self.logs_dir.exists())
        self.assertIn(["open", str(self.logs_dir)], opened)

    def test_update_context_triggers_restart_after_turn(self) -> None:
        self.obj.context_threshold_var = DummyVar("50")
        self.obj.process = object()
        self.obj.last_command = ["python", "gui.py"]
        with mock.patch.object(self.obj, "stop_supervisor") as stop:
            line = "[Builder] Turn completed (output tokens: 10) | context left: 100 tokens (~40.0%)"
            self.obj._update_context_from_line(line)
            stop.assert_not_called()
            self.assertTrue(self.obj._auto_restart_requested)
            wait_line = "[Supervisor] Turn 5: waiting for reviewer instructions..."
            self.obj._maybe_stop_after_current_turn(wait_line)
            stop.assert_called_once_with(auto=True)
            self.assertTrue(self.obj._auto_restart_pending)

    def test_update_context_ignored_without_threshold(self) -> None:
        self.obj.context_threshold_var = DummyVar("")
        self.obj.process = object()
        self.obj.last_command = ["python", "gui.py"]
        with mock.patch.object(self.obj, "stop_supervisor") as stop:
            self.obj._update_context_from_line(
                "[Reviewer] Turn completed | context left: 100 tokens (~40.0%)"
            )
            stop.assert_not_called()

    def test_sleep_prevention_mac_uses_caffeinate(self) -> None:
        self.obj.process = object()
        self.obj.prevent_screen_sleep_var = DummyBooleanVar(True)
        self.obj.prevent_computer_sleep_var = DummyBooleanVar(True)
        fake_proc = mock.Mock()
        fake_proc.wait.return_value = None
        original_platform = gui.sys.platform
        with mock.patch.object(gui.subprocess, "Popen", return_value=fake_proc) as popen:
            gui.sys.platform = "darwin"
            try:
                self.obj._start_sleep_prevention()
            finally:
                gui.sys.platform = original_platform
        popen.assert_called_once_with(["caffeinate", "-d", "-i"])
        self.assertIs(self.obj._sleep_process, fake_proc)
        self.obj._stop_sleep_prevention()
        fake_proc.terminate.assert_called_once()

    def test_sleep_prevention_noop_off_macos(self) -> None:
        self.obj.process = object()
        self.obj.prevent_screen_sleep_var = DummyBooleanVar(True)
        original_platform = gui.sys.platform
        with mock.patch.object(gui.subprocess, "Popen") as popen:
            gui.sys.platform = "linux"
            try:
                self.obj._start_sleep_prevention()
            finally:
                gui.sys.platform = original_platform
        popen.assert_not_called()
        self.assertIsNone(self.obj._sleep_process)


if __name__ == "__main__":
    unittest.main()
