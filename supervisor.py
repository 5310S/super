#!/usr/bin/env python3
"""Supervisor that coordinates Builder/Reviewer Codex agents with logging, config, and automation."""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol

DEFAULT_LOG_DIR = pathlib.Path.home() / ".codex-supervisor" / "logs"
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency is documented in README
    yaml = None

DEFAULT_BUILDER_PROMPT = (
    "You are the Builder agent. Execute commands, edit files, and implement the tasks provided by the Reviewer. "
    "Keep responses concise and focus on code and terminal output that the Reviewer needs."
)

DEFAULT_REVIEWER_PROMPT = (
    "You are the Reviewer agent. Inspect repository changes, craft precise prompts for the Builder, and keep a concise "
    "record of progress. Ask for clarification when needed and stop when the project goal is satisfied."
)

DEFAULT_REVIEWER_MARKER = "<<REVIEWER_DONE>>"
DEFAULT_BUILDER_MARKER = "<<BUILDER_DONE>>"
DEFAULT_COMMIT_TEMPLATE = "Supervisor turn {turn}: {summary}"
EXCERPT_MAX_BYTES = 200_000  # cap size of reviewer-requested excerpts (~200 KB)
BINARY_SNIFF_BYTES = 2048


@dataclass
class ReviewerTurn:
    summary: str
    prompt: str
    files: List[str]
    context: str


@dataclass
class ToolResult:
    name: str
    command: str
    exit_code: int
    duration_s: float
    output: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionState:
    last_turn: int
    latest_report: str
    latest_tool_outputs: str
    latest_reviewer_summary: str


class SessionRecorder:
    """Persists per-turn artifacts (JSONL/Markdown) plus optional git snapshots."""

    def __init__(
        self,
        base_dir: pathlib.Path,
        *,
        session_dir: Optional[pathlib.Path] = None,
        save_git_snapshots: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.utcnow().strftime("session-%Y%m%d-%H%M%S")
        self.session_dir = (session_dir or (self.base_dir / timestamp)).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.session_dir / "turns.jsonl"
        self.md_path = self.session_dir / "transcript.md"
        self.state_path = self.session_dir / "state.json"
        self.jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self.md = self.md_path.open("a", encoding="utf-8")
        self.save_git_snapshots = save_git_snapshots
        self._state = self._load_state()

    def _load_state(self) -> Optional[SessionState]:
        if not self.state_path.exists():
            return None
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return SessionState(
                last_turn=int(raw.get("last_turn", 0)),
                latest_report=raw.get("latest_report", "") or "",
                latest_tool_outputs=raw.get("latest_tool_outputs", "") or "",
                latest_reviewer_summary=raw.get("latest_reviewer_summary", "") or "",
            )
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    def get_state(self) -> Optional[SessionState]:
        return self._state

    def _write_state(self, state: SessionState) -> None:
        self._state = state
        payload = {
            "last_turn": state.last_turn,
            "latest_report": state.latest_report,
            "latest_tool_outputs": state.latest_tool_outputs,
            "latest_reviewer_summary": state.latest_reviewer_summary,
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _handle_record_error(self, state: SessionState, exc: OSError) -> None:
        print(f"[Supervisor] Failed to persist session artifacts: {exc}", file=sys.stderr, flush=True)
        self._state = state

    def record_turn(
        self,
        *,
        turn: int,
        repo_context: str,
        reviewer_turn: ReviewerTurn,
        builder_report: str,
        tool_results: List[ToolResult],
        tool_summary: str,
        git_snapshot: Optional[Dict[str, str]],
    ) -> None:
        state = SessionState(
            last_turn=turn,
            latest_report=builder_report or "",
            latest_tool_outputs=tool_summary,
            latest_reviewer_summary=reviewer_turn.summary or "",
        )
        try:
            payload = {
                "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                "turn": turn,
                "repo_context": repo_context,
                "reviewer": reviewer_turn.__dict__,
                "builder_report": builder_report,
                "tool_results": [result.as_dict() for result in tool_results],
                "tool_summary": tool_summary,
                "git_snapshot": git_snapshot,
            }
            self.jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.jsonl.flush()
            tool_block = "\n\n".join(
                f"### Tool `{res.name}` (exit {res.exit_code})\n{res.output}" for res in tool_results
            )
            snapshot_note = (
                f"Git snapshots saved to: status={git_snapshot['status']}, diff={git_snapshot['diff']}"
                if git_snapshot
                else "Git snapshots disabled or unavailable."
            )
            md_block = textwrap.dedent(
                f"""
                ## Turn {turn}
                **Reviewer Summary**\n{reviewer_turn.summary or 'N/A'}

                **Reviewer Prompt**\n{reviewer_turn.prompt}

                **Files Requested**\n{', '.join(reviewer_turn.files) or 'None'}

                **Builder Report**\n{builder_report or 'No builder response captured.'}

                **Repository Context**\n{repo_context}

                **Git Snapshot**\n{snapshot_note}

                **Tools**\n{tool_block or 'No tools executed.'}
                """
            ).strip()
            self.md.write(md_block + "\n\n")
            self.md.flush()
            self._write_state(state)
        except OSError as exc:
            self._handle_record_error(state, exc)

    def finalize(self, final_summary: Optional[str]) -> None:
        try:
            if final_summary:
                self.md.write(f"## Final Summary\n{final_summary}\n")
                self.md.flush()
        except OSError as exc:
            print(f"[Supervisor] Failed to finalize transcript: {exc}", file=sys.stderr, flush=True)
        finally:
            self.jsonl.close()
            self.md.close()

    def capture_git_artifacts(self, turn: int, repo_path: pathlib.Path) -> Optional[Dict[str, str]]:
        if not self.save_git_snapshots:
            return None
        status_path = self.session_dir / f"turn_{turn:03d}_status.txt"
        diff_path = self.session_dir / f"turn_{turn:03d}_diff.patch"
        try:
            status = run_git_command(["status", "-sb"], repo_path)
            diff = run_git_command(["diff"], repo_path)
        except FileNotFoundError as exc:
            message = f"Git executable not found: {exc}\n"
            status_path.write_text(message, encoding="utf-8")
            diff_path.write_text(message, encoding="utf-8")
            return {"status": str(status_path), "diff": str(diff_path)}
        except subprocess.CalledProcessError as exc:  # pragma: no cover - git failures surfaced to user
            status_path.write_text(f"Failed to capture git status: {exc}\n", encoding="utf-8")
            diff_path.write_text(f"Failed to capture git diff: {exc}\n", encoding="utf-8")
            return {"status": str(status_path), "diff": str(diff_path)}
        status_path.write_text(status, encoding="utf-8")
        diff_path.write_text(diff, encoding="utf-8")
        return {"status": str(status_path), "diff": str(diff_path)}


class ConfigError(RuntimeError):
    """Raised when configuration files are missing or malformed."""


class AgentInterface(Protocol):
    async def request(self, prompt: str) -> str:  # pragma: no cover - protocol definition
        ...


def parse_codex_exec_output(
    stream: str,
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> str:
    """Extract the final agent message from codex exec --json output."""
    if not stream:
        return ""
    messages: List[str] = []
    for line in stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if warn:
                warn(f"Failed to parse codex exec JSON line: {line[:160]}")
            continue
        if payload.get("type") != "item.completed":
            continue
        item = payload.get("item") or {}
        if item.get("type") == "agent_message":
            text = (item.get("text") or "").strip()
            if text:
                messages.append(text)
    return "\n\n".join(messages).strip() if messages else stream.strip()


def summarize_codex_event(
    line: str,
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        if warn:
            warn(f"Failed to parse codex exec event line: {line[:160]}")
        return None
    event_type = payload.get("type")
    if event_type == "thread.started":
        return "Thread started"
    if event_type == "turn.started":
        return "Turn started"
    if event_type == "turn.completed":
        usage = payload.get("usage") or {}
        tokens = usage.get("output_tokens")
        total = usage.get("total_tokens")
        context_window = (
            usage.get("context_window")
            or usage.get("context_limit")
            or payload.get("context_window")
        )
        summary = "Turn completed"
        if tokens is not None:
            summary += f" (output tokens: {tokens})"
        if context_window and total is not None:
            try:
                window_value = float(context_window)
                remaining = max(0.0, window_value - float(total))
                percent = (remaining / window_value) * 100 if window_value else 0.0
                summary += f" | context left: {int(remaining)} tokens (~{percent:.1f}%)"
            except (TypeError, ValueError):
                if warn:
                    warn(
                        "Failed to compute context usage from turn.completed event: "
                        f"context_window={context_window!r}, total_tokens={total!r}"
                    )
        return summary
    if event_type in {"item.started", "item.completed"}:
        item = payload.get("item") or {}
        item_type = item.get("type") or "item"
        if item_type == "reasoning":
            text = (item.get("text") or "").strip()
            if not text:
                return "Reasoning update"
            prefix = "Reasoning" if event_type == "item.completed" else "Reasoning (in progress)"
            return f"{prefix}: {text}"
        if item_type == "agent_message":
            text = (item.get("text") or "").strip()
            if not text:
                return "Agent message (empty)"
            return f"Agent message:\n{text}"
        if item_type == "command_execution":
            command = item.get("command") or ""
            status = item.get("status")
            exit_code = item.get("exit_code")
            output = (item.get("aggregated_output") or "").strip()
            verb = "Running" if event_type == "item.started" else "Command"
            summary = verb
            if command:
                summary += f": {command}"
            if status:
                summary += f" [{status}]"
            if exit_code is not None:
                summary += f" (exit {exit_code})"
            if output:
                summary += f"\n{output}"
            return summary
        return f"{event_type.replace('item.', '')} {item_type}"
    if warn:
        warn(f"Unhandled codex exec event: {line[:160]}")
    return None


class CodexExecAgent:
    """Launches codex exec per request to avoid the interactive TUI."""

    def __init__(
        self,
        *,
        name: str,
        role_prompt: str,
        command: str,
        extra_args: List[str],
        log_dir: pathlib.Path,
        repo_path: pathlib.Path,
        completion_marker: str,
        show_json: bool,
    ) -> None:
        self.name = name
        self.role_prompt = role_prompt.strip()
        self.command = command
        self.extra_args = extra_args
        self.repo_path = repo_path
        self.completion_marker = completion_marker
        self.show_json = show_json
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (log_dir / f"{name.lower()}_exec.log").open("a", encoding="utf-8")

    async def request(self, payload: str) -> str:
        prompt = "\n\n".join(part for part in (self.role_prompt, payload.strip()) if part).strip()
        return await asyncio.to_thread(self._run_once, prompt)

    def _run_once(self, prompt: str) -> str:
        cmd = [
            self.command,
            "exec",
            "--json",
            "--color",
            "never",
            *self.extra_args,
            prompt,
        ]
        self._log(
            "SUPERVISOR",
            f"Running codex exec ({self.name}) with extra args: {self.extra_args} | prompt preview: {prompt[:120]}",
        )
        print(f"[Supervisor] Launching codex exec for {self.name}...", flush=True)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.repo_path,
            )
        except FileNotFoundError as exc:
            message = f"Codex CLI not found while running {self.name}: {exc}"
            self._warn(message)
            raise RuntimeError(message) from exc
        stdout_lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            stripped = line.rstrip("\n")
            if stripped:
                self._log("JSON", stripped)
                if self.show_json:
                    print(f"[{self.name}][JSON] {stripped}", flush=True)
                else:
                    summary = summarize_codex_event(stripped, warn=self._warn)
                    if summary:
                        print(f"[{self.name}] {summary}", flush=True)
                    else:
                        print(f"[{self.name}] {stripped}", flush=True)
        proc.wait()
        stderr = ""
        if proc.stderr:
            stderr = proc.stderr.read()
            if stderr.strip():
                self._log("STDERR", stderr.strip())
                print(f"[{self.name}][STDERR] {stderr.strip()}", flush=True)
        if proc.returncode != 0:
            failure = stderr.strip() or "".join(stdout_lines).strip()
            message = f"{self.name} codex exec failed with exit code {proc.returncode}: {failure}"
            self._warn(message)
            raise RuntimeError(message)
        stdout = "".join(stdout_lines)
        message = parse_codex_exec_output(stdout, warn=self._warn)
        cleaned = self._strip_marker(message)
        if cleaned:
            self._log("STDOUT", cleaned)
        return cleaned

    def _strip_marker(self, text: str) -> str:
        marker = self.completion_marker.strip()
        if marker and marker in text:
            text = text.replace(marker, "")
        return text.strip()

    def _log(self, channel: str, message: str) -> None:
        timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
        self.log_file.write(f"{timestamp} [{channel}] {message}\n")
        self.log_file.flush()

    def _warn(self, message: str) -> None:
        self._log("WARN", message)
        print(f"[{self.name}][WARN] {message}", file=sys.stderr, flush=True)


def read_prompt(name: str, inline: Optional[str], path: Optional[str]) -> str:
    if path:
        prompt_path = pathlib.Path(path).expanduser()
        if not prompt_path.exists():
            raise FileNotFoundError(f"{name} prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    if inline:
        return inline
    return DEFAULT_BUILDER_PROMPT if name == "Builder" else DEFAULT_REVIEWER_PROMPT


def build_role_prompt(role: str, marker: str) -> str:
    if role == "Builder":
        instructions = (
            "You are the Builder agent. Execute commands, edit files, and implement the tasks provided by the Reviewer. "
            "Keep responses concise and focus on code and terminal output that the Reviewer needs."
        )
        marker_line = (
            f"Supervisor protocol: When you consider a task complete, summarize the work, then emit a line containing only {marker}. "
            "Do not add content after the marker."
        )
    else:
        instructions = (
            "You are the Reviewer agent. Inspect repository changes, craft precise prompts for the Builder, and keep a concise record of progress. "
            "Ask for clarification when needed and stop when the project goal is satisfied."
        )
        marker_line = (
            f"Supervisor protocol: After composing actionable guidance for the Builder, finish with a line containing only {marker}. "
            "If work is complete, set PROMPT to 'APPROVED' before the marker."
        )
    return f"{instructions}\n\n{marker_line}\n"


def build_reviewer_briefing(objective: str, marker: str) -> str:
    """Return a concise, ASCII-only reviewer briefing to avoid TUI crashes."""
    lines = [
        f"Supervisor Objective: {objective}",
        "Each response must follow this format:",
        "SUMMARY: 2-3 compact sentences referencing git status/diff/tests",
        "PROMPT: actionable instructions for the Builder or the word APPROVED",
        "FILES: optional space/comma separated file paths",
        "CONTEXT: optional extra notes",
        "",
        f"End every turn with a line containing only {marker}.",
        "If the project is complete, set PROMPT to APPROVED before that line.",
    ]
    return "\n".join(lines).strip()


def truncate_lines(text: str, max_lines: int) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... ({remaining} more lines truncated)"])


def run_git_command(args: List[str], repo_path: pathlib.Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def collect_repo_context(repo_path: pathlib.Path, status_lines: int, diff_lines: int) -> str:
    repo_path = repo_path.resolve()
    try:
        run_git_command(["rev-parse", "--show-toplevel"], repo_path)
    except subprocess.CalledProcessError:
        return "Git context unavailable (run inside a repository to enable status/diff summaries)."
    except FileNotFoundError:
        return "git executable not found."

    def _git_output(*args: str) -> str:
        try:
            return run_git_command(list(args), repo_path).strip()
        except subprocess.CalledProcessError as exc:
            return f"(Failed to collect {' '.join(args)}: {exc})"

    status = _git_output("status", "-sb") or "Working tree clean."
    diff = _git_output("diff", "--stat") or "No unstaged changes."
    status = truncate_lines(status, status_lines)
    diff = truncate_lines(diff, diff_lines)
    return f"Status:\n{status}\n\nDiff summary:\n{diff}"


def parse_reviewer_response(message: str) -> ReviewerTurn:
    sections = {"SUMMARY": [], "PROMPT": [], "FILES": [], "CONTEXT": []}
    current: Optional[str] = None
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                sections[current].append(line)
            continue
        header, sep, rest = stripped.partition(":")
        upper = header.upper()
        if upper in sections and sep:
            current = upper
            sections[current].append(rest.lstrip())
        else:
            if current:
                sections[current].append(line)
    summary = "\n".join(sections["SUMMARY"]).strip()
    prompt = "\n".join(sections["PROMPT"]).strip() or message.strip()
    file_blob = "\n".join(sections["FILES"])
    files = [frag.strip() for frag in re.split(r"[,\s]+", file_blob) if frag.strip()]
    context = "\n".join(sections["CONTEXT"]).strip()
    return ReviewerTurn(summary=summary, prompt=prompt, files=files, context=context)


def load_file_excerpt(repo_path: pathlib.Path, relative_path: str, max_lines: int) -> Tuple[str, str]:
    repo_root = repo_path.resolve()
    file_path = (repo_root / relative_path).resolve()
    try:
        file_path.relative_to(repo_root)
    except ValueError:
        return relative_path, "(Path escapes repository root.)"
    if not file_path.exists():
        return relative_path, "(File not found.)"
    if file_path.is_dir():
        return relative_path, "(Path is a directory.)"
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        return relative_path, f"(Failed to read file metadata: {exc})"
    if size > EXCERPT_MAX_BYTES:
        return relative_path, f"(File exceeds excerpt size limit of {EXCERPT_MAX_BYTES} bytes.)"
    try:
        with file_path.open("rb") as handle:
            chunk = handle.read(BINARY_SNIFF_BYTES)
    except OSError as exc:
        return relative_path, f"(Failed to read file: {exc})"
    if b"\x00" in chunk:
        return relative_path, "(Binary file omitted.)"
    text = file_path.read_text(encoding="utf-8", errors="replace")
    excerpt = truncate_lines(text, max_lines)
    return relative_path, excerpt or "(File is empty.)"


def derive_tool_name(command: str) -> str:
    return command.strip().split()[0] if command.strip() else "tool"


def run_tool_command(
    command: str,
    *,
    repo_path: pathlib.Path,
    timeout: int,
    output_lines: int,
) -> ToolResult:
    started = _dt.datetime.utcnow()
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = (completed.stdout + completed.stderr).strip()
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        combined = f"(Command timed out after {timeout}s)\n{(exc.stdout or '').strip()}\n{(exc.stderr or '').strip()}"
        exit_code = -1
    duration = (_dt.datetime.utcnow() - started).total_seconds()
    output = truncate_lines(combined, output_lines) or "(No output)"
    return ToolResult(
        name=derive_tool_name(command),
        command=command,
        exit_code=exit_code,
        duration_s=duration,
        output=output,
    )


def auto_commit_changes(repo_path: pathlib.Path, message: str) -> bool:
    try:
        status = run_git_command(["status", "--porcelain"], repo_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    if not status.strip():
        return False
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def format_commit_message(template: str, *, turn: Optional[int], summary: str, final_summary: str) -> str:
    context = {
        "turn": "final" if turn is None else turn,
        "summary": summary,
        "final_summary": final_summary,
    }
    return template.format(**context)


class ProtocolCoordinator:
    """Coordinates structured reviewer/builder turns and persistence."""

    def __init__(
        self,
        *,
        builder: AgentInterface,
        reviewer: AgentInterface,
        objective: str,
        reviewer_marker: str,
        builder_marker: str,
        max_turns: int,
        turn_timeout: float,
        repo_path: pathlib.Path,
        status_lines: int,
        diff_lines: int,
        file_excerpt_lines: int,
        builder_tools: List[str],
        builder_tool_timeout: int,
        tool_output_lines: int,
        recorder: SessionRecorder,
        auto_commit_each_turn: bool,
        auto_commit_final: bool,
        commit_template: str,
        initial_state: Optional[SessionState] = None,
    ) -> None:
        self.builder = builder
        self.reviewer = reviewer
        self.objective = objective.strip()
        self.reviewer_marker = reviewer_marker
        self.builder_marker = builder_marker
        self.max_turns = max_turns
        self.turn_timeout = turn_timeout
        self.repo_path = repo_path.resolve()
        self.status_lines = status_lines
        self.diff_lines = diff_lines
        self.file_excerpt_lines = file_excerpt_lines
        self.builder_tools = builder_tools
        self.builder_tool_timeout = builder_tool_timeout
        self.tool_output_lines = tool_output_lines
        self.recorder = recorder
        self.auto_commit_each_turn = auto_commit_each_turn
        self.auto_commit_final = auto_commit_final
        self.commit_template = commit_template
        self.initial_state = initial_state
        self.start_turn = (initial_state.last_turn + 1) if initial_state else 1
        self.reviewer_briefing = build_reviewer_briefing(self.objective, self.reviewer_marker)
        self.latest_report = (
            (initial_state.latest_report if initial_state else "") or "No builder output yet."
        )
        self.latest_tool_outputs = (
            (initial_state.latest_tool_outputs if initial_state else "") or "Tools not run yet."
        )
        self.latest_reviewer_summary = (
            (initial_state.latest_reviewer_summary if initial_state else "") or "No reviewer summary yet."
        )
        self.final_summary: Optional[str] = None

    async def run(self) -> None:
        try:
            end_turn = self.start_turn + self.max_turns
            for turn in range(self.start_turn, end_turn):
                print(f"[Supervisor] Turn {turn}: waiting for reviewer instructions...", flush=True)
                repo_context = collect_repo_context(self.repo_path, self.status_lines, self.diff_lines)
                reviewer_payload = textwrap.dedent(
                    f"""
                    {self.reviewer_briefing}

                    [Supervisor] Objective: {self.objective}
                    [Supervisor] Latest builder report:
                    {self.latest_report}

                    [Supervisor] Latest tool runs:
                    {self.latest_tool_outputs}

                    [Supervisor] Repository context:
                    {repo_context}
                    """
                ).strip()
                try:
                    reviewer_message = await self.reviewer.request(reviewer_payload)
                except RuntimeError as exc:
                    print(f"[Supervisor] Reviewer error: {exc}", flush=True)
                    self._record_turn_error(
                        turn=turn,
                        repo_context=repo_context,
                        error_message=f"Reviewer error: {exc}",
                        reviewer_turn=None,
                    )
                    break
                reviewer_turn = parse_reviewer_response(reviewer_message)
                self.latest_reviewer_summary = reviewer_turn.summary or "(Reviewer summary missing.)"
                if self._is_approval(reviewer_turn.prompt):
                    self.final_summary = self.latest_reviewer_summary or reviewer_turn.prompt
                    print("[Supervisor] Reviewer approved the work. Exiting protocol loop.", flush=True)
                    break
                file_excerpts = self._gather_file_excerpts(reviewer_turn.files)
                builder_payload = self._build_builder_payload(turn, reviewer_turn, file_excerpts)
                print("[Supervisor] Forwarding reviewer instructions to builder.", flush=True)
                try:
                    builder_report = await self.builder.request(builder_payload)
                except RuntimeError as exc:
                    print(f"[Supervisor] Builder error: {exc}", flush=True)
                    self._record_turn_error(
                        turn=turn,
                        repo_context=repo_context,
                        error_message=f"Builder error: {exc}",
                        reviewer_turn=reviewer_turn,
                    )
                    break
                self.latest_report = builder_report or "(Builder produced no summary.)"
                tool_summary, tool_results = await self._run_builder_tools()
                self.latest_tool_outputs = tool_summary
                snapshot = self.recorder.capture_git_artifacts(turn, self.repo_path)
                self.recorder.record_turn(
                    turn=turn,
                    repo_context=repo_context,
                    reviewer_turn=reviewer_turn,
                    builder_report=builder_report,
                    tool_results=tool_results,
                    tool_summary=tool_summary,
                    git_snapshot=snapshot,
                )
                if self.auto_commit_each_turn:
                    msg = format_commit_message(
                        self.commit_template,
                        turn=turn,
                        summary=self.latest_reviewer_summary,
                        final_summary=self.final_summary or "",
                    )
                    committed = auto_commit_changes(self.repo_path, msg)
                    if committed:
                        print(f"[Supervisor] Auto-committed turn {turn} changes.", flush=True)
            else:
                print("[Supervisor] Reached maximum turns without completion.", flush=True)
        finally:
            final_summary = self.final_summary or self.latest_reviewer_summary
            if self.auto_commit_final:
                msg = format_commit_message(
                    self.commit_template,
                    turn=None,
                    summary=self.latest_reviewer_summary,
                    final_summary=final_summary,
                )
                committed = auto_commit_changes(self.repo_path, msg)
                if committed:
                    print("[Supervisor] Auto-committed final changes.", flush=True)
            self.recorder.finalize(final_summary)

    async def _run_builder_tools(self) -> Tuple[str, List[ToolResult]]:
        if not self.builder_tools:
            return "No builder tools configured.", []
        results: List[ToolResult] = []
        for command in self.builder_tools:
            runner = asyncio.to_thread(
                run_tool_command,
                command,
                repo_path=self.repo_path,
                timeout=self.builder_tool_timeout,
                output_lines=self.tool_output_lines,
            )
            try:
                # Add a small buffer to allow subprocess shutdown/cleanup.
                result = await asyncio.wait_for(runner, timeout=self.builder_tool_timeout + 5)
            except asyncio.TimeoutError:
                warning = (
                    f"Supervisor timed out waiting for '{command}' after {self.builder_tool_timeout} seconds."
                )
                print(f"[Supervisor] {warning}", flush=True)
                result = ToolResult(
                    name=derive_tool_name(command),
                    command=command,
                    exit_code=-1,
                    duration_s=float(self.builder_tool_timeout),
                    output=warning,
                )
            results.append(result)
        summary_parts = [
            f"{res.name} (exit {res.exit_code}, {res.duration_s:.1f}s):\n{res.output}"
            for res in results
        ]
        summary = "\n\n".join(summary_parts) or "Builder tools produced no output."
        return summary, results

    def _gather_file_excerpts(self, files: List[str]) -> List[Tuple[str, str]]:
        excerpts: List[Tuple[str, str]] = []
        for rel_path in files:
            if not rel_path:
                continue
            excerpts.append(load_file_excerpt(self.repo_path, rel_path, self.file_excerpt_lines))
        return excerpts

    def _build_builder_payload(
        self,
        turn: int,
        reviewer_turn: ReviewerTurn,
        file_excerpts: List[Tuple[str, str]],
    ) -> str:
        parts = [
            f"[Reviewer -> Builder turn {turn}]",
            f"Summary:\n{reviewer_turn.summary or '(No summary provided)'}",
            f"Instructions:\n{reviewer_turn.prompt}",
        ]
        if reviewer_turn.context:
            parts.append(f"Additional context:\n{reviewer_turn.context}")
        if file_excerpts:
            excerpt_block = []
            for path, excerpt in file_excerpts:
                excerpt_block.append(f"--- {path} ---\n{excerpt}")
            parts.append("File excerpts:\n" + "\n\n".join(excerpt_block))
        parts.append(f"Latest supervisor tool output:\n{self.latest_tool_outputs}")
        parts.append(f"End with {self.builder_marker} once tasks are complete.")
        return "\n\n".join(parts)

    @staticmethod
    def _is_approval(prompt: str) -> bool:
        if not prompt:
            return False
        first_line = prompt.strip().splitlines()[0].strip().upper()
        return first_line.startswith("APPROVED") or first_line.startswith("DONE")

    def _record_turn_error(
        self,
        *,
        turn: int,
        repo_context: str,
        error_message: str,
        reviewer_turn: Optional[ReviewerTurn],
    ) -> None:
        placeholder = reviewer_turn or ReviewerTurn(
            summary="Reviewer turn failed before completion.",
            prompt=error_message,
            files=[],
            context="",
        )
        self.latest_reviewer_summary = placeholder.summary or "(Reviewer summary missing.)"
        self.latest_report = error_message
        self.latest_tool_outputs = error_message
        self.recorder.record_turn(
            turn=turn,
            repo_context=repo_context,
            reviewer_turn=placeholder,
            builder_report=error_message,
            tool_results=[],
            tool_summary=error_message,
            git_snapshot=None,
        )


def list_sessions(log_dir: pathlib.Path) -> List[pathlib.Path]:
    if not log_dir.exists():
        return []
    sessions = sorted(p for p in log_dir.iterdir() if p.is_dir())
    for path in sessions:
        print(path)
    return sessions


def resolve_session_path(log_dir: pathlib.Path, identifier: str) -> pathlib.Path:
    candidate = pathlib.Path(identifier)
    path = candidate if candidate.exists() else (log_dir / identifier)
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Session directory not found: {resolved}")
    return resolved


def show_session(log_dir: pathlib.Path, identifier: str) -> None:
    session_path = resolve_session_path(log_dir, identifier)
    transcript = session_path / "transcript.md"
    if not transcript.exists():
        raise FileNotFoundError(f"Transcript not found in {session_path}")
    print(transcript.read_text(encoding="utf-8"))


def load_config_data(path: pathlib.Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ConfigError("PyYAML is required for YAML configs. Install with `pip install pyyaml`. ")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping/dictionary.")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> None:
    valid_keys = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    filtered = {key: value for key, value in config.items() if key in valid_keys}
    unknown = set(config) - set(filtered)
    if unknown:
        print(f"[Supervisor] Ignoring unknown config keys: {', '.join(sorted(unknown))}")
    parser.set_defaults(**filtered)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Builder and Reviewer Codex CLI agents with shared logging.",
    )
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config file.")
    parser.add_argument("--codex-cli", default="codex", help="Path to the Codex CLI executable.")
    parser.add_argument("--builder-args", default="", help="Extra args for the builder Codex instance.")
    parser.add_argument("--reviewer-args", default="", help="Extra args for the reviewer Codex instance.")
    parser.add_argument(
        "--show-codex-json",
        action="store_true",
        help="Print raw JSON events from codex exec (otherwise print concise summaries).",
    )
    parser.add_argument("--builder-prompt", default=None, help="Inline role prompt for the builder agent.")
    parser.add_argument("--builder-prompt-file", default=None, help="Path to a file containing the builder prompt.")
    parser.add_argument("--reviewer-prompt", default=None, help="Inline role prompt for the reviewer agent.")
    parser.add_argument(
        "--reviewer-prompt-file",
        default=None,
        help="Path to a file containing the reviewer prompt.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help=f"Directory where per-session artifacts are stored (default: {DEFAULT_LOG_DIR}).",
    )
    parser.add_argument("--repo-path", default=".", help="Path to the repository the agents should operate on.")
    parser.add_argument("--auto-protocol", action="store_true", help="Enable structured Reviewer → Builder turns.")
    parser.add_argument("--objective", default=None, help="Shared project objective (required for auto protocol).")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum protocol turns to run.")
    parser.add_argument("--turn-timeout", type=int, default=300, help="Seconds to wait for each agent marker.")
    parser.add_argument("--reviewer-marker", default=DEFAULT_REVIEWER_MARKER, help="Reviewer turn completion marker.")
    parser.add_argument("--builder-marker", default=DEFAULT_BUILDER_MARKER, help="Builder turn completion marker.")
    parser.add_argument(
        "--context-status-lines",
        type=int,
        default=40,
        help="Maximum lines from `git status -sb` shared with Reviewer.",
    )
    parser.add_argument(
        "--context-diff-lines",
        type=int,
        default=120,
        help="Maximum lines from `git diff --stat` shared with Reviewer.",
    )
    parser.add_argument(
        "--file-excerpt-lines",
        type=int,
        default=120,
        help="Maximum lines per file excerpt when Reviewer requests FILES.",
    )
    parser.add_argument(
        "--builder-tool",
        action="append",
        default=[],
        help="Shell command to run after each builder turn (tests, lint, etc.).",
    )
    parser.add_argument(
        "--builder-tool-timeout",
        type=int,
        default=180,
        help="Seconds before an automatic builder tool command times out.",
    )
    parser.add_argument(
        "--tool-output-lines",
        type=int,
        default=120,
        help="Maximum lines of output to capture per builder tool.",
    )
    parser.add_argument(
        "--save-git-snapshots",
        dest="save_git_snapshots",
        action="store_true",
        default=True,
        help="Persist git status/diff per turn (default: enabled).",
    )
    parser.add_argument(
        "--no-save-git-snapshots",
        dest="save_git_snapshots",
        action="store_false",
        help="Disable git status/diff snapshots per turn.",
    )
    parser.add_argument(
        "--auto-commit-each-turn",
        action="store_true",
        help="Auto-commit repository changes after every builder turn.",
    )
    parser.add_argument(
        "--auto-commit-final",
        action="store_true",
        help="Auto-commit repository changes once reviewer approves.",
    )
    parser.add_argument(
        "--commit-template",
        default=DEFAULT_COMMIT_TEMPLATE,
        help="Template for auto-commit messages (placeholders: {turn}, {summary}, {final_summary}).",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List existing session transcripts in the log directory and exit.",
    )
    parser.add_argument(
        "--show-session",
        default=None,
        help="Display a previously recorded transcript (path or folder name) and exit.",
    )
    parser.add_argument(
        "--resume-session",
        default=None,
        help="Reuse an existing session directory instead of creating a new one.",
    )
    return parser


def resolve_codex_cli(cli: str) -> str:
    """Return an absolute path to the Codex CLI binary, ensuring it exists."""
    candidate = pathlib.Path(cli).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Codex CLI not found at {candidate}")
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(cli)
    if found:
        return found
    raise FileNotFoundError(f"Could not locate Codex CLI '{cli}'. Install it or pass --codex-cli /path/to/codex.")


async def main_async(args: argparse.Namespace) -> None:
    log_base = pathlib.Path(args.log_dir).expanduser().resolve()
    log_base.mkdir(parents=True, exist_ok=True)
    repo_path = pathlib.Path(args.repo_path).expanduser().resolve()

    if args.list_sessions:
        list_sessions(log_base)
        return
    if args.show_session:
        show_session(log_base, args.show_session)
        return

    builder_prompt = read_prompt("Builder", args.builder_prompt, args.builder_prompt_file)
    reviewer_prompt = read_prompt("Reviewer", args.reviewer_prompt, args.reviewer_prompt_file)
    if args.auto_protocol:
        builder_prompt = builder_prompt or ""
        reviewer_prompt = reviewer_prompt or ""
        builder_prompt = builder_prompt.strip()
        reviewer_prompt = reviewer_prompt.strip()
        builder_prompt = builder_prompt + "\n\n" + build_role_prompt("Builder", args.builder_marker)
        reviewer_prompt = reviewer_prompt + "\n\n" + build_role_prompt("Reviewer", args.reviewer_marker)

    builder = CodexExecAgent(
        name="Builder",
        role_prompt=builder_prompt,
        command=args.codex_cli,
        extra_args=shlex.split(args.builder_args),
        log_dir=log_base,
        repo_path=repo_path,
        completion_marker=args.builder_marker,
        show_json=args.show_codex_json,
    )
    reviewer = CodexExecAgent(
        name="Reviewer",
        role_prompt=reviewer_prompt,
        command=args.codex_cli,
        extra_args=shlex.split(args.reviewer_args),
        log_dir=log_base,
        repo_path=repo_path,
        completion_marker=args.reviewer_marker,
        show_json=args.show_codex_json,
    )

    if args.auto_protocol:
        session_dir = (
            resolve_session_path(log_base, args.resume_session) if args.resume_session else None
        )
        recorder = SessionRecorder(
            log_base,
            session_dir=session_dir,
            save_git_snapshots=args.save_git_snapshots,
        )
        coordinator = ProtocolCoordinator(
            builder=builder,
            reviewer=reviewer,
            objective=args.objective or "",
            reviewer_marker=args.reviewer_marker,
            builder_marker=args.builder_marker,
            max_turns=args.max_turns,
            turn_timeout=args.turn_timeout,
            repo_path=repo_path,
            status_lines=args.context_status_lines,
            diff_lines=args.context_diff_lines,
            file_excerpt_lines=args.file_excerpt_lines,
            builder_tools=args.builder_tool,
            builder_tool_timeout=args.builder_tool_timeout,
            tool_output_lines=args.tool_output_lines,
            recorder=recorder,
            auto_commit_each_turn=args.auto_commit_each_turn,
            auto_commit_final=args.auto_commit_final,
            commit_template=args.commit_template,
            initial_state=recorder.get_state(),
        )
        await coordinator.run()
    else:  # pragma: no cover - guarded by CLI parsing
        raise RuntimeError("Non protocol mode is not supported in exec-backed supervisor.")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.config:
        config_data = load_config_data(pathlib.Path(args.config))
        apply_config_defaults(parser, config_data)
        args = parser.parse_args()
    if args.auto_protocol and not args.objective:
        parser.error("--objective is required when --auto-protocol is enabled.")
    if not args.auto_protocol:
        parser.error("Headless supervisor currently requires --auto-protocol.")
    try:
        args.codex_cli = resolve_codex_cli(args.codex_cli)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nSupervisor interrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
