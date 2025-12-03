#!/usr/bin/env python3
"""Simple macOS-friendly GUI wrapper for launching supervisor sessions."""

from __future__ import annotations

import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from shutil import which
from typing import Any


APP_TITLE = "Codex Supervisor"
CLI_SENTINEL = "--__supervisor_cli__"
LOGS_DIR = pathlib.Path.home() / ".codex-supervisor" / "logs"
SETTINGS_PATH = pathlib.Path.home() / ".codex-supervisor" / "gui-settings.json"
OUTPUT_MAX_CHARS = 80_000
PROCESS_TERMINATE_TIMEOUT = 5  # seconds before escalating from terminate -> kill
CONTEXT_LINE_RE = re.compile(
    r"\[(Builder|Reviewer)\].*context left: ([0-9.]+) tokens(?: \(~([0-9.]+)%\))?",
    re.IGNORECASE,
)
TURN_WAIT_RE = re.compile(r"^\[Supervisor\] Turn \d+: waiting for reviewer instructions", re.IGNORECASE)


def default_codex_path() -> str:
    """Return the best-guess codex binary path."""

    candidates = []
    found = which("codex")
    if found:
        return found
    candidates.append(pathlib.Path.home() / "codex" / "bin" / "codex")
    candidates.append(pathlib.Path("/opt/homebrew/bin/codex"))
    candidates.append(pathlib.Path("/usr/local/bin/codex"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "codex"


def _maybe_run_cli() -> None:
    """If invoked with the sentinel flag, run the supervisor CLI and exit."""

    if CLI_SENTINEL not in sys.argv:
        return
    idx = sys.argv.index(CLI_SENTINEL)
    cli_args = sys.argv[idx + 1 :]
    sys.argv = ["codex-supervisor", *cli_args]
    from supervisor import main as supervisor_main

    supervisor_main()
    sys.exit(0)


_maybe_run_cli()


class SupervisorTab(tk.Frame):
    """A single supervisor instance hosted inside the GUI notebook."""

    def __init__(
        self,
        master: tk.Misc,
        controller: "SupervisorGUI",
        tab_id: str,
        title: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(master)
        self.controller = controller
        self.tab_id = tab_id
        self.title = title
        self.process: subprocess.Popen | None = None
        self.output_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._auto_restart_requested = False
        self._auto_restart_pending = False
        self._restart_reason = ""
        self._user_stop_requested = False
        self._stop_after_prompt_requested = False
        self.last_command: list[str] | None = None
        self._sleep_process: subprocess.Popen | None = None
        self._sleep_warning_shown = False
        self._context_stats: dict[str, dict[str, float] | None] = {"Builder": None, "Reviewer": None}
        self.timer_var = tk.StringVar(value="00:00:00")
        self.carousel_rotations_var = tk.IntVar(value=0)
        self._timer_start: float | None = None
        self._timer_job: str | None = None
        self._drain_job: str | None = None
        self._settings_vars: dict[str, tk.Variable] = {}
        self._destroyed = False
        self.config_var = tk.StringVar()
        self.objective_var = tk.StringVar()
        self.repo_var = tk.StringVar(value=str(pathlib.Path.cwd()))
        self.codex_cli_var = tk.StringVar(value=default_codex_path())
        self.auto_protocol_var = tk.BooleanVar(value=True)
        self.auto_commit_var = tk.BooleanVar(value=True)
        self.auto_push_var = tk.BooleanVar(value=False)
        self.builder_args_var = tk.StringVar()
        self.reviewer_args_var = tk.StringVar()
        self.show_json_var = tk.BooleanVar(value=False)
        self.context_threshold_var = tk.StringVar()
        self.carousel_var = tk.BooleanVar(value=False)
        self.prevent_screen_sleep_var = tk.BooleanVar(value=False)
        self.prevent_computer_sleep_var = tk.BooleanVar(value=False)
        self._register_setting("config_path", self.config_var)
        self._register_setting("objective", self.objective_var)
        self._register_setting("repo_path", self.repo_var)
        self._register_setting("codex_cli", self.codex_cli_var)
        self._register_setting("auto_protocol", self.auto_protocol_var)
        self._register_setting("auto_commit_final", self.auto_commit_var)
        self._register_setting("auto_push_final", self.auto_push_var)
        self._register_setting("builder_args", self.builder_args_var)
        self._register_setting("reviewer_args", self.reviewer_args_var)
        self._register_setting("show_json", self.show_json_var)
        self._register_setting("context_threshold_percent", self.context_threshold_var)
        self._register_setting("carousel", self.carousel_var)
        self._register_setting("prevent_screen_sleep", self.prevent_screen_sleep_var)
        self._register_setting("prevent_computer_sleep", self.prevent_computer_sleep_var)
        self._build_widgets()
        if state:
            self._apply_state(state)
        self._schedule_drain()

    # ------------------------------------------------------------------
    # Widget + state helpers
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        config_frame = tk.LabelFrame(self, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(config_frame, text="Config file (optional):").grid(row=0, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.config_var, width=60).grid(row=0, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Browse", command=self._browse_config).grid(row=0, column=2, padx=5)

        tk.Label(config_frame, text="Objective:").grid(row=1, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.objective_var, width=60).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="we",
            padx=5,
        )

        tk.Label(config_frame, text="Repo path:").grid(row=2, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.repo_var, width=60).grid(row=2, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Choose", command=self._browse_repo).grid(row=2, column=2, padx=5)

        tk.Label(config_frame, text="Codex CLI path:").grid(row=3, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.codex_cli_var, width=60).grid(row=3, column=1, sticky="we", padx=5)

        tk.Checkbutton(config_frame, text="Enable auto protocol", variable=self.auto_protocol_var).grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )

        tk.Checkbutton(
            config_frame,
            text="Auto commit when reviewer approves",
            variable=self.auto_commit_var,
        ).grid(row=4, column=1, sticky="w", pady=(5, 0))
        self.auto_push_button = tk.Button(
            config_frame,
            text="Commit + push when reviewer approves: Off",
            command=self._toggle_auto_push,
        )
        self.auto_push_button.grid(row=4, column=2, sticky="w", pady=(5, 0), padx=5)

        tk.Label(config_frame, text="Builder extra args:").grid(row=5, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.builder_args_var, width=60).grid(
            row=5,
            column=1,
            columnspan=2,
            sticky="we",
            padx=5,
        )

        tk.Label(config_frame, text="Reviewer extra args:").grid(row=6, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.reviewer_args_var, width=60).grid(
            row=6,
            column=1,
            columnspan=2,
            sticky="we",
            padx=5,
        )

        tk.Checkbutton(
            config_frame,
            text="Show raw Codex JSON events",
            variable=self.show_json_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))

        tk.Label(config_frame, text="Auto restart if context below (%):").grid(row=8, column=0, sticky="w")
        tk.Entry(config_frame, textvariable=self.context_threshold_var, width=20).grid(
            row=8,
            column=1,
            sticky="w",
            padx=5,
        )

        tk.Checkbutton(
            config_frame,
            text="Carousel: restart automatically after exit code 0",
            variable=self.carousel_var,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(5, 0))

        tk.Checkbutton(
            config_frame,
            text="Keep screen awake while supervisor runs",
            variable=self.prevent_screen_sleep_var,
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))

        tk.Checkbutton(
            config_frame,
            text="Keep computer awake even if display sleeps",
            variable=self.prevent_computer_sleep_var,
        ).grid(row=11, column=0, columnspan=2, sticky="w")

        config_frame.columnconfigure(1, weight=1)
        self.auto_push_var.trace_add("write", lambda *_: self._refresh_auto_push_button())
        self._refresh_auto_push_button()

        controls = tk.Frame(self)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.start_button = tk.Button(controls, text="Start Supervisor", command=self.start_supervisor)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(controls, text="Stop", command=self.stop_supervisor, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_after_prompt_button = tk.Button(
            controls,
            text=self._default_graceful_label(),
            command=self.stop_after_prompt,
            state=tk.DISABLED,
        )
        self.stop_after_prompt_button.pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Open Logs Folder", command=self.controller.open_logs_folder).pack(side=tk.LEFT, padx=5)

        stats_frame = tk.Frame(controls)
        stats_frame.pack(side=tk.RIGHT)

        rotations_frame = tk.Frame(stats_frame)
        rotations_frame.pack(side=tk.LEFT, padx=(0, 15))
        tk.Label(rotations_frame, text="Carousel Rotations:").pack(side=tk.LEFT)
        tk.Label(rotations_frame, textvariable=self.carousel_rotations_var).pack(side=tk.LEFT, padx=(2, 0))

        timer_frame = tk.Frame(stats_frame)
        timer_frame.pack(side=tk.LEFT)
        tk.Label(timer_frame, text="Elapsed:").pack(side=tk.LEFT)
        tk.Label(timer_frame, textvariable=self.timer_var).pack(side=tk.LEFT, padx=(2, 0))

        output_frame = tk.LabelFrame(self, text="Supervisor Output", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text["yscrollcommand"] = scrollbar.set
        self._reset_graceful_stop_state()

    def _toggle_auto_push(self) -> None:
        new_value = not bool(self.auto_push_var.get())
        self.auto_push_var.set(new_value)
        if new_value and not self.auto_commit_var.get():
            self.auto_commit_var.set(True)

    def _refresh_auto_push_button(self) -> None:
        state = "On" if self.auto_push_var.get() else "Off"
        self.auto_push_button.config(text=f"Commit + push when reviewer approves: {state}")

    def _graceful_shortcut_hint(self) -> str:
        return "Cmd+G" if sys.platform == "darwin" else "Ctrl+G"

    def _default_graceful_label(self) -> str:
        return f"Stop Gracefully ({self._graceful_shortcut_hint()})"

    def _register_setting(self, key: str, var: tk.Variable) -> None:
        self._settings_vars[key] = var
        var.trace_add("write", lambda *_: self.controller.schedule_save())

    def _apply_state(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            var = self._settings_vars.get(key)
            if var is None:
                continue
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(value)

    def get_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, var in self._settings_vars.items():
            data[key] = var.get()
        return data

    def rename(self, title: str) -> None:
        self.title = title

    def _schedule_drain(self) -> None:
        if self._destroyed:
            return
        self._drain_job = self.after(100, self._drain_output_queue)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")]
        )
        if path:
            self.config_var.set(path)

    def _browse_repo(self) -> None:
        path = filedialog.askdirectory(initialdir=self.repo_var.get() or os.getcwd())
        if path:
            self.repo_var.set(path)

    # ------------------------------------------------------------------
    # Supervisor control + logging
    # ------------------------------------------------------------------
    def start_supervisor(self) -> None:
        if self.process:
            messagebox.showwarning(APP_TITLE, "Supervisor is already running in this tab.")
            return
        self.controller.clear_attention(self)
        cmd = self._build_command()
        if not cmd:
            return
        self._auto_restart_pending = False
        self._restart_reason = ""
        self._user_stop_requested = False
        self._reset_graceful_stop_state()
        self._reset_carousel_rotations()
        self._launch_supervisor(cmd, remember=True)

    def stop_after_prompt(self) -> None:
        if not self.process:
            messagebox.showinfo(APP_TITLE, "No supervisor is running in this tab.")
            return
        if self._stop_after_prompt_requested:
            messagebox.showinfo(APP_TITLE, "A graceful stop is already scheduled for this tab.")
            return
        self._stop_after_prompt_requested = True
        self._auto_restart_requested = False
        self._auto_restart_pending = False
        self._restart_reason = ""
        self._log_line("\nWill stop after the current prompt finishes...\n")
        self.stop_after_prompt_button.config(text="Graceful stop scheduled")

    def _build_command(self) -> list[str] | None:
        cmd = [sys.executable, CLI_SENTINEL]
        config_path = self.config_var.get().strip()
        if config_path:
            cmd.extend(["--config", config_path])
        else:
            objective = self.objective_var.get().strip()
            if self.auto_protocol_var.get() and not objective:
                messagebox.showwarning(APP_TITLE, "Objective is required when auto protocol is enabled.")
                return None
            if objective:
                cmd.extend(["--objective", objective])
            repo_path = self.repo_var.get().strip() or "."
            cmd.extend(["--repo-path", repo_path])
        if self.auto_protocol_var.get():
            cmd.append("--auto-protocol")
        auto_push_on_approval = self.auto_push_var.get()
        if self.auto_commit_var.get() or auto_push_on_approval:
            cmd.append("--auto-commit-final")
        if auto_push_on_approval:
            cmd.append("--auto-push-final")
        try:
            cli_value = self._resolve_codex_cli(self.codex_cli_var.get())
        except (FileNotFoundError, PermissionError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return None
        if cli_value:
            cmd.extend(["--codex-cli", cli_value])
        builder_args = self.builder_args_var.get().strip()
        if builder_args:
            cmd.extend(["--builder-args", builder_args])
        reviewer_args = self.reviewer_args_var.get().strip()
        if reviewer_args:
            cmd.extend(["--reviewer-args", reviewer_args])
        if self.show_json_var.get():
            cmd.append("--show-codex-json")
        return cmd

    def _launch_supervisor(self, cmd: list[str], *, remember: bool) -> None:
        if remember:
            self.last_command = list(cmd)
        self._log_line(f"Launching: {' '.join(cmd)}\n")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Failed to start supervisor: {exc}")
            self.process = None
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_after_prompt_button.config(state=tk.NORMAL, text=self._default_graceful_label())
        if self.process.stdout:
            threading.Thread(target=self._reader_thread, args=(self.process.stdout, "STDOUT"), daemon=True).start()
        if self.process.stderr:
            threading.Thread(target=self._reader_thread, args=(self.process.stderr, "STDERR"), daemon=True).start()
        self.after(500, self._check_process)
        self._start_sleep_prevention()
        self._start_timer()

    def stop_supervisor(self, *, auto: bool = False) -> None:
        self.stop_after_prompt_button.config(state=tk.DISABLED, text=self._default_graceful_label())
        self._stop_after_prompt_requested = False
        if not self.process:
            self._stop_sleep_prevention()
            return
        self._user_stop_requested = not auto
        if auto:
            self._auto_restart_pending = True
        else:
            self._auto_restart_requested = False
            self._auto_restart_pending = False
            self._restart_reason = ""
        self._log_line("\nStopping supervisor...\n")
        self._stop_sleep_prevention()
        try:
            self.process.terminate()
        except OSError as exc:
            self._log_line(f"\nFailed to terminate supervisor: {exc}\n")
        threading.Thread(target=self._wait_for_exit, args=(self.process,), daemon=True).start()

    def _reader_thread(self, stream, channel: str) -> None:
        for line in stream:
            self.output_queue.put((channel, line))
        stream.close()

    def _drain_output_queue(self) -> None:
        while not self.output_queue.empty():
            channel, line = self.output_queue.get()
            prefix = "" if channel == "STDOUT" else "[STDERR] "
            self._log_line(prefix + line)
        self._schedule_drain()

    def _log_line(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        try:
            total_chars = int(self.output_text.count("1.0", "end-1c", "chars")[0])
        except tk.TclError:
            total_chars = 0
        if total_chars > OUTPUT_MAX_CHARS:
            excess = total_chars - OUTPUT_MAX_CHARS
            self.output_text.delete("1.0", f"1.0+{excess}c")
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)
        for line in text.splitlines():
            self._handle_log_line(line)

    def _start_sleep_prevention(self) -> None:
        if self._sleep_process or not self.process:
            return
        keep_screen = bool(self.prevent_screen_sleep_var.get())
        keep_system = bool(self.prevent_computer_sleep_var.get())
        if not (keep_screen or keep_system):
            return
        if sys.platform != "darwin":
            if not self._sleep_warning_shown:
                self._sleep_warning_shown = True
                self._log_line("\nSleep prevention requires macOS (caffeinate). Options ignored.\n")
            return
        args = ["caffeinate"]
        if keep_screen:
            args.append("-d")
        if keep_system:
            args.append("-i")
        try:
            self._sleep_process = subprocess.Popen(args)
        except OSError as exc:
            self._sleep_process = None
            self._log_line(f"\nFailed to enable sleep prevention: {exc}\n")
            return
        self._log_line(f"\nSleep prevention enabled ({' '.join(args)}).\n")

    def _stop_sleep_prevention(self) -> None:
        proc = self._sleep_process
        if not proc:
            return
        self._sleep_process = None
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
        except OSError:
            pass
        self._log_line("\nSleep prevention disabled.\n")

    def _start_timer(self) -> None:
        if self._timer_job:
            try:
                self.after_cancel(self._timer_job)
            except tk.TclError:
                pass
            self._timer_job = None
        self._timer_start = time.monotonic()
        self.timer_var.set("00:00:00")
        self._schedule_timer_tick()

    def _schedule_timer_tick(self) -> None:
        if self._timer_start is None or self._destroyed:
            self._timer_job = None
            self.timer_var.set("00:00:00")
            return
        elapsed = max(0.0, time.monotonic() - self._timer_start)
        total_seconds = int(elapsed)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self._timer_job = self.after(1000, self._schedule_timer_tick)

    def _stop_timer(self) -> None:
        if self._timer_job:
            try:
                self.after_cancel(self._timer_job)
            except tk.TclError:
                pass
            self._timer_job = None
        self._timer_start = None
        self.timer_var.set("00:00:00")

    def _reset_carousel_rotations(self) -> None:
        self.carousel_rotations_var.set(0)

    def _increment_carousel_rotations(self) -> None:
        try:
            current = int(self.carousel_rotations_var.get())
        except (TypeError, ValueError):
            current = 0
        self.carousel_rotations_var.set(current + 1)

    def _handle_log_line(self, line: str) -> None:
        self._update_context_from_line(line)
        self._maybe_stop_after_current_turn(line)

    def _context_threshold_percent(self) -> float | None:
        raw = self.context_threshold_var.get().strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return max(0.0, value)

    def _update_context_from_line(self, line: str) -> None:
        match = CONTEXT_LINE_RE.search(line)
        if not match:
            return
        agent = match.group(1).capitalize()
        tokens = float(match.group(2))
        percent_text = match.group(3)
        percent = float(percent_text) if percent_text is not None else None
        self._context_stats[agent] = {"tokens": tokens, "percent": percent}
        threshold = self._context_threshold_percent()
        if threshold is None or percent is None:
            return
        if percent <= threshold:
            self._initiate_auto_restart(agent, percent)

    def _initiate_auto_restart(self, agent: str, percent: float) -> None:
        if (
            self._auto_restart_pending
            or self._auto_restart_requested
            or not self.process
            or not self.last_command
        ):
            return
        self._auto_restart_requested = True
        self._restart_reason = f"{agent} context low ({percent:.1f}%)"
        self._log_line(f"\nAuto restart scheduled after current turn: {self._restart_reason}\n")

    def _maybe_stop_after_current_turn(self, line: str) -> None:
        if not self.process:
            return
        if not TURN_WAIT_RE.search(line):
            return
        if self._stop_after_prompt_requested:
            self._reset_graceful_stop_state()
            self._log_line("\nCurrent prompt finished; stopping this tab as requested...\n")
            self.stop_supervisor()
            return
        if self._auto_restart_requested and self.last_command:
            self._auto_restart_requested = False
            self._auto_restart_pending = True
            self.stop_supervisor(auto=True)

    def _wait_for_exit(self, proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._log_line("\nSupervisor unresponsive; killing process...\n")
            try:
                proc.kill()
            except OSError:
                pass

    def _resolve_codex_cli(self, value: str) -> str:
        candidate = (value or "").strip() or default_codex_path()
        path = pathlib.Path(candidate).expanduser()
        if path.exists():
            if path.is_dir():
                raise FileNotFoundError(f"Codex CLI path is a directory: {path}")
            if not os.access(path, os.X_OK):
                raise PermissionError(f"Codex CLI is not executable: {path}")
            return str(path)
        found = which(candidate)
        if found:
            return found
        raise FileNotFoundError(f"Could not locate Codex CLI '{candidate}'. Provide an absolute path in the GUI.")

    def _restart_supervisor(self) -> None:
        if not self.last_command:
            self._log_line("\nNo previous supervisor command to restart.\n")
            return
        reason = self._restart_reason or "context threshold reached"
        self._log_line(f"\nRestarting supervisor ({reason})...\n")
        self._launch_supervisor(self.last_command, remember=False)
        self._auto_restart_requested = False
        self._auto_restart_pending = False
        self._restart_reason = ""

    def _check_process(self) -> None:
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            manual_stop = self._user_stop_requested
            self._user_stop_requested = False
            self._log_line(f"\nSupervisor exited with code {code}\n")
            self.process = None
            self._stop_sleep_prevention()
            self._stop_timer()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.stop_after_prompt_button.config(state=tk.DISABLED, text=self._default_graceful_label())
            self._reset_graceful_stop_state()
            if self._auto_restart_requested and self.last_command:
                self._auto_restart_pending = True
                self._auto_restart_requested = False
            if self._auto_restart_pending and self.last_command:
                self._auto_restart_pending = False
                self.after(500, self._restart_supervisor)
                return
            carousel_enabled = bool(self.carousel_var.get())
            if carousel_enabled and code == 0 and self.last_command and not manual_stop and not self._destroyed:
                self._increment_carousel_rotations()
                self._restart_reason = "carousel enabled"
                self.after(500, self._restart_supervisor)
                return
            if code not in (0, None):
                messagebox.showwarning(APP_TITLE, f"Supervisor exited with code {code}")
            if not self._destroyed:
                self.controller.tab_finished(self)
            return
        if self.process and not self._destroyed:
            self.after(500, self._check_process)

    # ------------------------------------------------------------------
    # Tab lifecycle helpers
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def prepare_for_close(self) -> bool:
        if self.is_running():
            messagebox.showwarning(APP_TITLE, "Stop this supervisor before closing the tab.")
            return False
        return True

    def shutdown(self) -> None:
        self._destroyed = True
        if self._drain_job:
            try:
                self.after_cancel(self._drain_job)
            except tk.TclError:
                pass
            self._drain_job = None
        self._stop_timer()
        self._stop_sleep_prevention()

    def _reset_graceful_stop_state(self) -> None:
        self._stop_after_prompt_requested = False
        try:
            self.stop_after_prompt_button.config(text=self._default_graceful_label())
        except tk.TclError:
            pass


class SupervisorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x600")
        self.minsize(700, 400)
        self.settings_path = SETTINGS_PATH
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_pending = False
        self._loading_settings = False
        self.tabs: dict[str, SupervisorTab] = {}
        self._next_tab_index = 1
        self._attention_states: dict[str, dict[str, Any]] = {}
        self._gatekeeper_help_dismissed = False
        self._gatekeeper_help_shown = False
        self._build_widgets()
        self._load_settings()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after_idle(self._maybe_show_gatekeeper_help)

    def _build_widgets(self) -> None:
        toolbar = tk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 0))
        tk.Button(toolbar, text="New Tab", command=self.add_tab).pack(side=tk.LEFT)
        self.close_tab_button = tk.Button(toolbar, text="Close Tab", command=self.close_current_tab)
        self.close_tab_button.pack(side=tk.LEFT, padx=5)
        self.rename_tab_button = tk.Button(toolbar, text="Rename Tab", command=self.rename_current_tab)
        self.rename_tab_button.pack(side=tk.LEFT)
        tk.Button(toolbar, text="Gatekeeper Help", command=self._force_gatekeeper_help).pack(side=tk.LEFT, padx=10)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _bind_shortcuts(self) -> None:
        modifier = "Command" if sys.platform == "darwin" else "Control"
        self.bind_all(f"<{modifier}-g>", self._stop_gracefully_shortcut)

    def _stop_gracefully_shortcut(self, event: tk.Event | None = None) -> str:
        tab = self._current_tab()
        if tab:
            tab.stop_after_prompt()
        return "break"

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------
    def _default_tab_title(self) -> str:
        count = len(self.tabs) + 1
        return f"Session {count}"

    def _generate_tab_id(self) -> str:
        tab_id = f"tab-{self._next_tab_index}"
        self._next_tab_index += 1
        return tab_id

    def _reserve_tab_id(self, tab_id: str) -> None:
        match = re.fullmatch(r"tab-(\d+)", tab_id)
        if not match:
            return
        value = int(match.group(1)) + 1
        if value > self._next_tab_index:
            self._next_tab_index = value

    def add_tab(
        self,
        *,
        tab_id: str | None = None,
        title: str | None = None,
        state: dict[str, Any] | None = None,
        select: bool = True,
    ) -> str:
        if tab_id is None:
            tab_id = self._generate_tab_id()
        self._reserve_tab_id(tab_id)
        if title is None:
            title = self._default_tab_title()
        tab = SupervisorTab(self.notebook, self, tab_id, title, state=state)
        self.tabs[tab_id] = tab
        self.notebook.add(tab, text=title)
        if select:
            self.notebook.select(tab)
        self._update_tab_buttons()
        if not self._loading_settings:
            self.schedule_save()
        return tab_id

    def _current_tab(self) -> SupervisorTab | None:
        widget_name = self.notebook.select()
        if not widget_name:
            return None
        widget = self.nametowidget(widget_name)
        if isinstance(widget, SupervisorTab):
            return widget
        return None

    def close_current_tab(self) -> None:
        tab = self._current_tab()
        if not tab:
            return
        if len(self.tabs) <= 1:
            messagebox.showinfo(APP_TITLE, "At least one session tab is required.")
            return
        if not tab.prepare_for_close():
            return
        self._remove_tab(tab)

    def _remove_tab(self, tab: SupervisorTab) -> None:
        self.clear_attention(tab)
        tab.shutdown()
        self.notebook.forget(tab)
        self.tabs.pop(tab.tab_id, None)
        tab.destroy()
        if not self.tabs:
            self.add_tab()
        else:
            self._update_tab_buttons()
            if not self._loading_settings:
                self.schedule_save()

    def rename_current_tab(self) -> None:
        tab = self._current_tab()
        if not tab:
            return
        current_title = self.notebook.tab(tab, "text")
        new_title = simpledialog.askstring(APP_TITLE, "Session name:", initialvalue=current_title, parent=self)
        if not new_title:
            return
        trimmed = new_title.strip()
        if not trimmed:
            return
        tab.rename(trimmed)
        self.notebook.tab(tab, text=trimmed)
        self.schedule_save()

    def _update_tab_buttons(self) -> None:
        disable_close = len(self.tabs) <= 1
        state = tk.DISABLED if disable_close else tk.NORMAL
        self.close_tab_button.config(state=state)
        rename_state = tk.NORMAL if self.tabs else tk.DISABLED
        self.rename_tab_button.config(state=rename_state)

    def _on_tab_changed(self, event: tk.Event | None = None) -> None:
        self._update_tab_buttons()
        tab = self._current_tab()
        if tab:
            self.clear_attention(tab)

    def tab_finished(self, tab: SupervisorTab) -> None:
        self.request_attention(tab)

    def request_attention(self, tab: SupervisorTab) -> None:
        if tab.tab_id in self._attention_states:
            return
        if not tab.winfo_exists():
            return
        if self._current_tab() is tab:
            return
        base_title = self.notebook.tab(tab, "text")
        state: dict[str, Any] = {"base": base_title, "flash": False, "job": None}
        self._attention_states[tab.tab_id] = state
        self._flash_tab(tab)

    def _flash_tab(self, tab: SupervisorTab) -> None:
        state = self._attention_states.get(tab.tab_id)
        if not state:
            return
        if not tab.winfo_exists():
            self.clear_attention(tab)
            return
        state["flash"] = not state.get("flash", False)
        title = state["base"]
        if state["flash"]:
            title = f"*** {title} ***"
        self.notebook.tab(tab, text=title)
        state["job"] = self.after(600, self._flash_tab, tab)

    def clear_attention(self, tab: SupervisorTab) -> None:
        state = self._attention_states.pop(tab.tab_id, None)
        if not state:
            return
        job = state.get("job")
        if job:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        if tab.winfo_exists():
            self.notebook.tab(tab, text=state["base"])

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def schedule_save(self) -> None:
        if self._loading_settings or self._save_pending:
            return
        self._save_pending = True
        self.after(500, self._save_settings)

    def _collect_tabs_state(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for widget_name in self.notebook.tabs():
            widget = self.nametowidget(widget_name)
            if not isinstance(widget, SupervisorTab):
                continue
            data.append(
                {
                    "id": widget.tab_id,
                    "title": self.notebook.tab(widget, "text"),
                    "settings": widget.get_state(),
                }
            )
        return data

    def _save_settings(self) -> None:
        self._save_pending = False
        data = {
            "tabs": self._collect_tabs_state(),
            "active_tab": None,
            "next_tab_index": self._next_tab_index,
            "gatekeeper_help_dismissed": self._gatekeeper_help_dismissed,
        }
        active_widget = self.notebook.select()
        if active_widget:
            widget = self.nametowidget(active_widget)
            if isinstance(widget, SupervisorTab):
                data["active_tab"] = widget.tab_id
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Failed to save settings: {exc}")

    def _load_settings(self) -> None:
        if not self.settings_path.exists():
            self.add_tab()
            return
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.add_tab()
            return
        self._loading_settings = True
        try:
            stored_next = raw.get("next_tab_index")
            if isinstance(stored_next, int) and stored_next > 0:
                self._next_tab_index = stored_next
            if isinstance(raw.get("gatekeeper_help_dismissed"), bool):
                self._gatekeeper_help_dismissed = bool(raw["gatekeeper_help_dismissed"])
            tabs_data = raw.get("tabs") or []
            for tab_data in tabs_data:
                tab_id = tab_data.get("id") or self._generate_tab_id()
                title = tab_data.get("title") or self._default_tab_title()
                settings = tab_data.get("settings") or {}
                self.add_tab(tab_id=tab_id, title=title, state=settings, select=False)
            if not self.tabs:
                self.add_tab()
            active_id = raw.get("active_tab")
            if active_id and active_id in self.tabs:
                self.notebook.select(self.tabs[active_id])
        finally:
            self._loading_settings = False
        self._update_tab_buttons()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def open_logs_folder(self) -> None:
        log_dir = LOGS_DIR
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Failed to prepare log directory: {exc}")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(log_dir)], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(log_dir)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(log_dir)], check=False)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Failed to open log directory: {exc}")

    def _maybe_show_gatekeeper_help(self) -> None:
        if sys.platform != "darwin":
            return
        if self._gatekeeper_help_shown:
            return
        if self._gatekeeper_help_dismissed:
            return
        self._gatekeeper_help_shown = True
        self._show_gatekeeper_popup(force=False)

    def _force_gatekeeper_help(self) -> None:
        if sys.platform != "darwin":
            messagebox.showinfo(APP_TITLE, "Gatekeeper help is only relevant on macOS.")
            return
        self._show_gatekeeper_popup(force=True)

    def _show_gatekeeper_popup(self, *, force: bool) -> None:
        popup = tk.Toplevel(self)
        popup.title("Open Anyway on macOS")
        popup.geometry("520x360")
        popup.grab_set()
        popup.transient(self)

        message = (
            "macOS Gatekeeper blocks unsigned apps. If you see\n"
            "\"Apple cannot verify the app is free of malware\":\n\n"
            "1) Open System Settings → Privacy & Security.\n"
            "2) Scroll to Security and click \"Open Anyway\".\n"
            "3) Confirm and reopen the app.\n\n"
            "Or, from Terminal (once) to remove the quarantine flag:\n"
            "xattr -dr com.apple.quarantine /Applications/Codex\\ Supervisor.app"
        )
        tk.Message(popup, text=message, width=480, justify="left").pack(padx=15, pady=15, anchor="w")

        cmd = "xattr -dr com.apple.quarantine /Applications/Codex\\ Supervisor.app"
        buttons = tk.Frame(popup)
        buttons.pack(fill=tk.X, padx=15, pady=(0, 10))

        def copy_cmd() -> None:
            try:
                popup.clipboard_clear()
                popup.clipboard_append(cmd)
            except tk.TclError:
                return
            messagebox.showinfo(APP_TITLE, "Command copied. Paste into Terminal and run.")

        tk.Button(buttons, text="Copy Terminal command", command=copy_cmd).pack(side=tk.LEFT)
        dont_show_var = tk.BooleanVar(value=False)
        tk.Checkbutton(buttons, text="Don't show again", variable=dont_show_var).pack(side=tk.RIGHT)

        def close_popup() -> None:
            if not self._gatekeeper_help_dismissed or force:
                self._gatekeeper_help_dismissed = bool(dont_show_var.get())
                if self._gatekeeper_help_dismissed:
                    self.schedule_save()
            popup.destroy()

        tk.Button(popup, text="Got it", command=close_popup).pack(pady=(0, 12))
        popup.protocol("WM_DELETE_WINDOW", close_popup)

    def _on_close(self) -> None:
        running_tabs = [tab for tab in self.tabs.values() if tab.is_running()]
        if running_tabs:
            if not messagebox.askyesno(APP_TITLE, "Supervisor sessions are still running. Stop all and exit?"):
                return
            for tab in running_tabs:
                tab.stop_supervisor()
            self.after(100, self._wait_for_all_tabs)
            return
        self.destroy()

    def _wait_for_all_tabs(self) -> None:
        if any(tab.is_running() for tab in self.tabs.values()):
            self.after(100, self._wait_for_all_tabs)
            return
        self.destroy()


def main() -> None:
    app = SupervisorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
