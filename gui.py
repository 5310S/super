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
import tkinter as tk
from tkinter import filedialog, messagebox
from shutil import which


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


class SupervisorGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x600")
        self.minsize(700, 400)
        self.process: subprocess.Popen | None = None
        self.output_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._closing_window = False
        self._save_pending = False
        self._loading_settings = False
        self.settings_path = SETTINGS_PATH
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._auto_restart_requested = False
        self._context_stats = {"Builder": None, "Reviewer": None}
        self._auto_restart_pending = False
        self._restart_reason = ""
        self.last_command: list[str] | None = None
        self._settings_vars: dict[str, tk.Variable] = {}
        self._build_widgets()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)

    def _build_widgets(self) -> None:
        config_frame = tk.LabelFrame(self, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(config_frame, text="Config file (optional):").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar()
        self._register_setting("config_path", self.config_var)
        tk.Entry(config_frame, textvariable=self.config_var, width=60).grid(row=0, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Browse", command=self._browse_config).grid(row=0, column=2, padx=5)

        tk.Label(config_frame, text="Objective:").grid(row=1, column=0, sticky="w")
        self.objective_var = tk.StringVar()
        self._register_setting("objective", self.objective_var)
        tk.Entry(config_frame, textvariable=self.objective_var, width=60).grid(
            row=1, column=1, columnspan=2, sticky="we", padx=5
        )

        tk.Label(config_frame, text="Repo path:").grid(row=2, column=0, sticky="w")
        self.repo_var = tk.StringVar(value=str(pathlib.Path.cwd()))
        self._register_setting("repo_path", self.repo_var)
        tk.Entry(config_frame, textvariable=self.repo_var, width=60).grid(row=2, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Choose", command=self._browse_repo).grid(row=2, column=2, padx=5)

        tk.Label(config_frame, text="Codex CLI path:").grid(row=3, column=0, sticky="w")
        self.codex_cli_var = tk.StringVar(value=default_codex_path())
        self._register_setting("codex_cli", self.codex_cli_var)
        tk.Entry(config_frame, textvariable=self.codex_cli_var, width=60).grid(row=3, column=1, sticky="we", padx=5)

        self.auto_protocol_var = tk.BooleanVar(value=True)
        self._register_setting("auto_protocol", self.auto_protocol_var)
        tk.Checkbutton(config_frame, text="Enable auto protocol", variable=self.auto_protocol_var).grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )

        self.auto_commit_var = tk.BooleanVar(value=True)
        self._register_setting("auto_commit_final", self.auto_commit_var)
        tk.Checkbutton(
            config_frame,
            text="Auto commit when reviewer approves",
            variable=self.auto_commit_var,
        ).grid(row=4, column=1, sticky="w", pady=(5, 0))

        tk.Label(config_frame, text="Builder extra args:").grid(row=5, column=0, sticky="w")
        self.builder_args_var = tk.StringVar()
        self._register_setting("builder_args", self.builder_args_var)
        tk.Entry(config_frame, textvariable=self.builder_args_var, width=60).grid(
            row=5, column=1, columnspan=2, sticky="we", padx=5
        )

        tk.Label(config_frame, text="Reviewer extra args:").grid(row=6, column=0, sticky="w")
        self.reviewer_args_var = tk.StringVar()
        self._register_setting("reviewer_args", self.reviewer_args_var)
        tk.Entry(config_frame, textvariable=self.reviewer_args_var, width=60).grid(
            row=6, column=1, columnspan=2, sticky="we", padx=5
        )

        self.show_json_var = tk.BooleanVar(value=False)
        self._register_setting("show_json", self.show_json_var)
        tk.Checkbutton(
            config_frame,
            text="Show raw Codex JSON events",
            variable=self.show_json_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))

        tk.Label(config_frame, text="Auto restart if context below (%):").grid(row=8, column=0, sticky="w")
        self.context_threshold_var = tk.StringVar()
        self._register_setting("context_threshold_percent", self.context_threshold_var)
        tk.Entry(config_frame, textvariable=self.context_threshold_var, width=20).grid(
            row=8, column=1, sticky="w", padx=5
        )

        config_frame.columnconfigure(1, weight=1)

        controls = tk.Frame(self)
        controls.pack(fill=tk.X, padx=10)

        self.start_button = tk.Button(controls, text="Start Supervisor", command=self.start_supervisor)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(controls, text="Stop", command=self.stop_supervisor, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Open Logs Folder", command=self._open_logs).pack(side=tk.LEFT, padx=5)

        output_frame = tk.LabelFrame(self, text="Supervisor Output", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text["yscrollcommand"] = scrollbar.set

    def _register_setting(self, key: str, var: tk.Variable) -> None:
        self._settings_vars[key] = var
        var.trace_add("write", lambda *_: self._schedule_save())

    def _schedule_save(self) -> None:
        if self._loading_settings or self._save_pending:
            return
        self._save_pending = True
        self.after(500, self._save_settings)

    def _save_settings(self) -> None:
        self._save_pending = False
        data = {key: var.get() for key, var in self._settings_vars.items()}
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            self._log_line(f"\nFailed to save settings: {exc}\n")

    def _load_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._loading_settings = True
        try:
            for key, var in self._settings_vars.items():
                if key not in raw:
                    continue
                value = raw[key]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                else:
                    var.set(value)
        finally:
            self._loading_settings = False

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")])
        if path:
            self.config_var.set(path)

    def _browse_repo(self) -> None:
        path = filedialog.askdirectory(initialdir=self.repo_var.get() or os.getcwd())
        if path:
            self.repo_var.set(path)

    def _open_logs(self) -> None:
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

    def start_supervisor(self) -> None:
        if self.process:
            messagebox.showwarning(APP_TITLE, "Supervisor is already running.")
            return
        cmd = self._build_command()
        if not cmd:
            return
        self._auto_restart_pending = False
        self._restart_reason = ""
        self._launch_supervisor(cmd, remember=True)

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
        if self.auto_commit_var.get():
            cmd.append("--auto-commit-final")
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
        if self.process.stdout:
            threading.Thread(
                target=self._reader_thread,
                args=(self.process.stdout, "STDOUT"),
                daemon=True,
            ).start()
        if self.process.stderr:
            threading.Thread(
                target=self._reader_thread,
                args=(self.process.stderr, "STDERR"),
                daemon=True,
            ).start()
        self.after(500, self._check_process)

    def stop_supervisor(self, *, wait: bool = False, auto: bool = False) -> None:
        if not self.process:
            return
        if auto:
            self._auto_restart_pending = True
        else:
            self._auto_restart_requested = False
            self._auto_restart_pending = False
            self._restart_reason = ""
        self._log_line("\nStopping supervisor...\n")
        try:
            self.process.terminate()
        except OSError as exc:
            self._log_line(f"\nFailed to terminate supervisor: {exc}\n")
        threading.Thread(
            target=self._wait_for_exit,
            args=(self.process, wait),
            daemon=True,
        ).start()

    def _reader_thread(self, stream, channel: str) -> None:
        for line in stream:
            self.output_queue.put((channel, line))
        stream.close()

    def _drain_output_queue(self) -> None:
        while True:
            try:
                channel, line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            prefix = "" if channel == "STDOUT" else "[STDERR] "
            self._log_line(prefix + line)
        self.after(100, self._drain_output_queue)

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
        if (
            not self._auto_restart_requested
            or not self.process
            or not self.last_command
        ):
            return
        if TURN_WAIT_RE.search(line):
            self._auto_restart_requested = False
            self._auto_restart_pending = True
            self.stop_supervisor(auto=True)

    def _wait_for_exit(self, proc: subprocess.Popen, wait_for_close: bool) -> None:
        try:
            proc.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._log_line("\nSupervisor unresponsive; killing process...\n")
            try:
                proc.kill()
            except OSError:
                pass
        if wait_for_close:
            self.after(0, self._wait_for_close)

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno(APP_TITLE, "Supervisor is still running. Stop it and exit?"):
                return
            self._closing_window = True
            self._auto_restart_requested = False
            self._auto_restart_pending = False
            self.stop_supervisor(wait=True)
            return
        self.destroy()

    def _wait_for_close(self) -> None:
        if self.process and self.process.poll() is None:
            self.after(100, self._wait_for_close)
            return
        self.destroy()

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
            self._log_line(f"\nSupervisor exited with code {code}\n")
            self.process = None
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if self._auto_restart_requested and self.last_command:
                self._auto_restart_pending = True
                self._auto_restart_requested = False
            if self._auto_restart_pending and self.last_command:
                self._auto_restart_pending = False
                self.after(500, self._restart_supervisor)
                return
            if code not in (0, None):
                messagebox.showwarning(APP_TITLE, f"Supervisor exited with code {code}")
            return
        if self.process:
            self.after(500, self._check_process)


def main() -> None:
    app = SupervisorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
