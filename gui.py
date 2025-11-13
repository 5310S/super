#!/usr/bin/env python3
"""Simple macOS-friendly GUI wrapper for launching supervisor sessions."""

from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox


APP_TITLE = "Codex Supervisor"
CLI_SENTINEL = "--__supervisor_cli__"


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
        self.output_queue: queue.Queue[str] = queue.Queue()
        self._build_widgets()
        self.after(100, self._drain_output_queue)

    def _build_widgets(self) -> None:
        config_frame = tk.LabelFrame(self, text="Configuration", padx=10, pady=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(config_frame, text="Config file (optional):").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar()
        tk.Entry(config_frame, textvariable=self.config_var, width=60).grid(row=0, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Browse", command=self._browse_config).grid(row=0, column=2, padx=5)

        tk.Label(config_frame, text="Objective:").grid(row=1, column=0, sticky="w")
        self.objective_var = tk.StringVar()
        tk.Entry(config_frame, textvariable=self.objective_var, width=60).grid(
            row=1, column=1, columnspan=2, sticky="we", padx=5
        )

        tk.Label(config_frame, text="Repo path:").grid(row=2, column=0, sticky="w")
        self.repo_var = tk.StringVar(value=str(pathlib.Path.cwd()))
        tk.Entry(config_frame, textvariable=self.repo_var, width=60).grid(row=2, column=1, sticky="we", padx=5)
        tk.Button(config_frame, text="Choose", command=self._browse_repo).grid(row=2, column=2, padx=5)

        self.auto_protocol_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="Enable auto protocol", variable=self.auto_protocol_var).grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )

        self.auto_commit_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            config_frame,
            text="Auto commit when reviewer approves",
            variable=self.auto_commit_var,
        ).grid(row=3, column=1, sticky="w", pady=(5, 0))

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

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Config files", "*.yaml *.yml *.json"), ("All files", "*.*")])
        if path:
            self.config_var.set(path)

    def _browse_repo(self) -> None:
        path = filedialog.askdirectory(initialdir=self.repo_var.get() or os.getcwd())
        if path:
            self.repo_var.set(path)

    def _open_logs(self) -> None:
        log_dir = pathlib.Path(self.repo_var.get() or ".").expanduser().resolve() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["open", str(log_dir)])
        elif sys.platform.startswith("win"):
            os.startfile(log_dir)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(log_dir)])

    def start_supervisor(self) -> None:
        if self.process:
            messagebox.showwarning(APP_TITLE, "Supervisor is already running.")
            return
        cmd = [sys.executable, CLI_SENTINEL]
        config_path = self.config_var.get().strip()
        if config_path:
            cmd.extend(["--config", config_path])
        else:
            objective = self.objective_var.get().strip()
            if self.auto_protocol_var.get() and not objective:
                messagebox.showwarning(APP_TITLE, "Objective is required when auto protocol is enabled.")
                return
            if objective:
                cmd.extend(["--objective", objective])
            repo_path = self.repo_var.get().strip() or "."
            cmd.extend(["--repo-path", repo_path])
            if self.auto_protocol_var.get():
                cmd.append("--auto-protocol")
            if self.auto_commit_var.get():
                cmd.append("--auto-commit-final")

        self._log_line(f"Launching: {' '.join(cmd)}\n")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
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
        threading.Thread(target=self._reader_thread, daemon=True).start()
        self.after(500, self._check_process)

    def stop_supervisor(self) -> None:
        if not self.process:
            return
        self.process.terminate()
        self._log_line("\nStopping supervisor...\n")

    def _reader_thread(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.output_queue.put(line)
        self.process.stdout.close()

    def _drain_output_queue(self) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._log_line(line)
        self.after(100, self._drain_output_queue)

    def _log_line(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _check_process(self) -> None:
        if self.process and self.process.poll() is not None:
            code = self.process.returncode
            self._log_line(f"\nSupervisor exited with code {code}\n")
            self.process = None
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
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
