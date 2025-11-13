#!/usr/bin/env python3
"""Lightweight supervisor that orchestrates two Codex CLI agents."""

import argparse
import asyncio
import datetime as _dt
import pathlib
import shlex
import subprocess
import sys
from contextlib import suppress
from typing import List, Optional, Tuple


DEFAULT_BUILDER_PROMPT = """\
You are the Builder agent. Execute commands, edit files, and implement the \
tasks provided by the Reviewer. Keep responses concise and focus on code and \
terminal output that the Reviewer needs.\
"""

DEFAULT_REVIEWER_PROMPT = """\
You are the Reviewer agent. Inspect repository changes, craft precise prompts \
for the Builder, and keep a concise record of progress. Ask for clarification \
when needed and stop when the project goal is satisfied.\
"""

DEFAULT_REVIEWER_MARKER = "<<REVIEWER_DONE>>"
DEFAULT_BUILDER_MARKER = "<<BUILDER_DONE>>"


def read_prompt(name: str, inline: Optional[str], path: Optional[str]) -> str:
    """Return the prompt text from either a string or a file."""
    if path:
        prompt_path = pathlib.Path(path).expanduser()
        if not prompt_path.exists():
            raise FileNotFoundError(f"{name} prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    if inline:
        return inline
    return DEFAULT_BUILDER_PROMPT if name == "Builder" else DEFAULT_REVIEWER_PROMPT


def add_marker_instruction(base_prompt: str, marker: str, role: str) -> str:
    """Append protocol marker instructions to the base prompt text."""
    cleaned = base_prompt.rstrip()
    if role == "Builder":
        instruction = (
            f"Supervisor protocol: When you consider a task complete, summarize the work, "
            f"then emit a line containing only {marker}. Do not add content after the marker."
        )
    else:
        instruction = (
            f"Supervisor protocol: After composing actionable guidance for the Builder, "
            f"finish with a line containing only {marker}. If work is complete, write DONE on "
            "its own line right before the marker."
        )
    return f"{cleaned}\n\n{instruction}\n"


def truncate_lines(text: str, max_lines: int) -> str:
    """Trim text to a maximum number of lines with an ellipsis indicator."""
    if not text:
        return ""
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... ({remaining} more lines truncated)"])


def collect_repo_context(status_lines: int = 40, diff_lines: int = 120) -> str:
    """Return a compact git status/diff summary, or a fallback message."""
    cwd = pathlib.Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Git context unavailable (run inside a repository to enable status/diff summaries)."

    repo_root = result.stdout.strip()

    def _git_output(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return out
        except subprocess.CalledProcessError as exc:
            return f"(Failed to collect {' '.join(args)}: {exc})"

    status = _git_output("status", "-sb") or "Working tree clean."
    diff = _git_output("diff", "--stat") or "No unstaged changes."
    status = truncate_lines(status, status_lines)
    diff = truncate_lines(diff, diff_lines)
    return f"Repo root: {repo_root}\nStatus:\n{status}\n\nDiff summary:\n{diff}"


def parse_reviewer_response(message: str) -> Tuple[str, str]:
    """Extract SUMMARY and PROMPT blocks from the reviewer output."""
    summary_lines: List[str] = []
    prompt_lines: List[str] = []
    current: Optional[str] = None
    for line in message.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("SUMMARY:"):
            fragment = line.split(":", 1)[1].lstrip() if ":" in line else ""
            summary_lines.append(fragment)
            current = "summary"
            continue
        if upper.startswith("PROMPT:"):
            fragment = line.split(":", 1)[1].lstrip() if ":" in line else ""
            prompt_lines.append(fragment)
            current = "prompt"
            continue
        if current == "summary":
            summary_lines.append(line)
        elif current == "prompt":
            prompt_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    prompt = "\n".join(prompt_lines).strip()
    if not prompt:
        prompt = message.strip()
    return summary, prompt


class AgentSession:
    """Thin wrapper around a Codex CLI subprocess."""

    def __init__(
        self,
        name: str,
        role_prompt: str,
        command: str,
        extra_args: List[str],
        log_dir: pathlib.Path,
    ) -> None:
        self.name = name
        self.role_prompt = role_prompt.strip() + "\n"
        self.command = command
        self.extra_args = extra_args
        self.log_dir = log_dir
        self.process: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self.stdout_queue: Optional[asyncio.Queue[str]] = None
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (log_dir / f"{name.lower()}.log").open("a", encoding="utf-8")

    async def start(self) -> None:
        """Start the Codex CLI process for this agent."""
        argv = [self.command, *self.extra_args]
        self._log_meta("SUPERVISOR", f"Starting: {' '.join(shlex.quote(a) for a in argv)}")
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.stdout_queue = asyncio.Queue()
        self._stdout_task = asyncio.create_task(self._pump_stream(self.process.stdout, "STDOUT"))
        self._stderr_task = asyncio.create_task(self._pump_stream(self.process.stderr, "STDERR"))
        await self.send(self.role_prompt)

    async def _pump_stream(self, stream: asyncio.StreamReader, label: str) -> None:
        """Read a subprocess stream and log/echo each line."""
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode(errors="replace")
            self._log_meta(label, decoded.rstrip("\n"))
            prefix = f"[{self.name}][{label}] "
            sys.stdout.write(prefix + decoded)
            sys.stdout.flush()
            if label == "STDOUT" and self.stdout_queue is not None:
                await self.stdout_queue.put(decoded)

    async def send(self, text: str) -> None:
        """Send text to the subprocess stdin."""
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"{self.name} process is not running")
        self.process.stdin.write(text.encode())
        await self.process.stdin.drain()
        for line in text.rstrip("\n").splitlines():
            self._log_meta("PROMPT", line)

    async def stop(self) -> None:
        """Terminate the process if it is still running."""
        if not self.process:
            return
        if self.process.returncode is None:
            self._log_meta("SUPERVISOR", "Terminating process")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._log_meta("SUPERVISOR", "Force killing process")
                self.process.kill()
        await self._close_stream_tasks()
        self.stdout_queue = None
        self.log_file.close()

    async def _close_stream_tasks(self) -> None:
        for task in (self._stdout_task, self._stderr_task):
            if not task:
                continue
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    def _log_meta(self, channel: str, message: str) -> None:
        timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
        self.log_file.write(f"{timestamp} [{channel}] {message}\n")
        self.log_file.flush()

    async def read_until_marker(self, marker: str, timeout: Optional[float] = None) -> str:
        """Collect stdout until a dedicated marker line is observed."""
        if self.stdout_queue is None:
            raise RuntimeError(f"{self.name} session is not initialized for marker reads")
        buffer: List[str] = []
        try:
            while True:
                line = await asyncio.wait_for(self.stdout_queue.get(), timeout)
                stripped = line.strip()
                if stripped == marker:
                    return "".join(buffer).strip()
                buffer.append(line)
        except asyncio.TimeoutError as exc:
            raise MarkerTimeout(f"{self.name} did not emit marker {marker} in time.") from exc


class MarkerTimeout(RuntimeError):
    """Raised when an agent fails to emit its required marker in time."""
    pass


async def interactive_loop(builder: AgentSession, reviewer: AgentSession) -> None:
    """Simple REPL to relay supervisor input to the selected agent."""
    print(
        "\nSupervisor REPL ready. Prefix input with 'b:' or 'r:' (or 'both:'). "
        "Use 'quit' or Ctrl+D to exit.\n",
        flush=True,
    )
    loop = asyncio.get_running_loop()
    while True:
        try:
            raw = await loop.run_in_executor(None, sys.stdin.readline)
        except KeyboardInterrupt:
            break
        if not raw:
            break
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"quit", "exit"}:
            break
        target, payload = parse_repl_input(line)
        if not payload:
            print("No payload detected; use e.g. 'b: run tests'", flush=True)
            continue
        if target == "builder":
            await builder.send(payload)
        elif target == "reviewer":
            await reviewer.send(payload)
        else:
            await builder.send(payload)
            await reviewer.send(payload)


def parse_repl_input(line: str) -> (str, str):
    """Parse REPL input to determine target agent and payload."""
    lowered = line.lower()
    for prefix, name in (("b:", "builder"), ("r:", "reviewer"), ("both:", "both"), ("all:", "both")):
        if lowered.startswith(prefix):
            payload = line[len(prefix) :].lstrip()
            return name, payload
    return "builder", line


class ProtocolCoordinator:
    """Coordinates structured turn-taking between Reviewer and Builder."""

    def __init__(
        self,
        *,
        builder: AgentSession,
        reviewer: AgentSession,
        objective: str,
        reviewer_marker: str,
        builder_marker: str,
        max_turns: int,
        turn_timeout: float,
    ) -> None:
        self.builder = builder
        self.reviewer = reviewer
        self.objective = objective.strip()
        self.reviewer_marker = reviewer_marker
        self.builder_marker = builder_marker
        self.max_turns = max_turns
        self.turn_timeout = turn_timeout
        self.latest_report = "No builder output yet."
        self.latest_reviewer_summary = "No reviewer summary yet."
        self.final_summary: Optional[str] = None

    async def run(self) -> None:
        """Drive Reviewer → Builder turns until completion or limits hit."""
        await self._prime_reviewer()
        for turn in range(1, self.max_turns + 1):
            print(f"[Supervisor] Turn {turn}: waiting for reviewer instructions...", flush=True)
            repo_context = collect_repo_context()
            await self.reviewer.send(
                f"\n[Supervisor] Objective: {self.objective}\n"
                f"[Supervisor] Latest builder report:\n{self.latest_report}\n\n"
                f"[Supervisor] Repository context:\n{repo_context}\n"
                f"Respond using the SUMMARY/PROMPT format and end with {self.reviewer_marker}."
            )
            try:
                reviewer_message = await self.reviewer.read_until_marker(
                    self.reviewer_marker, timeout=self.turn_timeout
                )
            except MarkerTimeout as exc:
                print(f"[Supervisor] Reviewer timeout: {exc}", flush=True)
                break
            reviewer_message = reviewer_message.strip()
            summary, prompt = parse_reviewer_response(reviewer_message)
            self.latest_reviewer_summary = summary or "(Reviewer did not provide a summary.)"
            if self._is_approval(prompt):
                self.final_summary = self.latest_reviewer_summary or prompt
                print("[Supervisor] Reviewer approved the work. Exiting protocol loop.", flush=True)
                break
            print(f"[Supervisor] Forwarding reviewer instructions to builder.", flush=True)
            await self.builder.send(
                f"[Reviewer -> Builder turn {turn}]\n"
                f"Summary:\n{self.latest_reviewer_summary}\n\n"
                f"Instructions:\n{prompt}\n"
                f"Carry out the work, summarize results, and end with {self.builder_marker}."
            )
            try:
                self.latest_report = await self.builder.read_until_marker(
                    self.builder_marker, timeout=self.turn_timeout
                )
            except MarkerTimeout as exc:
                print(f"[Supervisor] Builder timeout: {exc}", flush=True)
                break
        else:
            print("[Supervisor] Reached maximum turns without completion.", flush=True)

    async def _prime_reviewer(self) -> None:
        briefing = (
            f"Supervisor Objective: {self.objective}\n"
            "Each response must follow this format:\n"
            "SUMMARY: <2-3 compact sentences referencing git status/diff/tests>\n"
            "PROMPT: <actionable instructions for the Builder or 'APPROVED'>\n\n"
            f"End every turn with a line containing only {self.reviewer_marker}. "
            "If the project is complete, set PROMPT to 'APPROVED' (or 'DONE') right before the marker."
        )
        await self.reviewer.send(briefing)

    @staticmethod
    def _is_approval(prompt: str) -> bool:
        """Return True if the reviewer indicates the project is complete."""
        if not prompt:
            return False
        first_line = prompt.strip().splitlines()[0].strip().upper()
        return first_line.startswith("APPROVED") or first_line.startswith("DONE")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Builder and Reviewer Codex CLI agents with shared logging."
    )
    parser.add_argument("--codex-cli", default="codex", help="Path to the Codex CLI executable.")
    parser.add_argument(
        "--builder-args",
        default="",
        help="Extra args for the builder Codex instance (quoted string).",
    )
    parser.add_argument(
        "--reviewer-args",
        default="",
        help="Extra args for the reviewer Codex instance (quoted string).",
    )
    parser.add_argument(
        "--builder-prompt",
        default=None,
        help="Inline role prompt for the builder agent.",
    )
    parser.add_argument(
        "--builder-prompt-file",
        default=None,
        help="Path to a file containing the builder prompt.",
    )
    parser.add_argument(
        "--reviewer-prompt",
        default=None,
        help="Inline role prompt for the reviewer agent.",
    )
    parser.add_argument(
        "--reviewer-prompt-file",
        default=None,
        help="Path to a file containing the reviewer prompt.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/supervisor",
        help="Directory where per-agent logs will be written.",
    )
    parser.add_argument(
        "--auto-protocol",
        action="store_true",
        help="Enable structured Reviewer → Builder turn-taking.",
    )
    parser.add_argument(
        "--objective",
        default=None,
        help="Shared project objective (required when --auto-protocol is set).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum Reviewer → Builder cycles to run in protocol mode.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=int,
        default=300,
        help="Seconds to wait for each agent to emit its marker in protocol mode.",
    )
    parser.add_argument(
        "--reviewer-marker",
        default=DEFAULT_REVIEWER_MARKER,
        help="Line marker the Reviewer must emit to finish a turn.",
    )
    parser.add_argument(
        "--builder-marker",
        default=DEFAULT_BUILDER_MARKER,
        help="Line marker the Builder must emit to finish a turn.",
    )
    return parser


async def main_async(args: argparse.Namespace) -> None:
    log_dir = pathlib.Path(args.log_dir).expanduser()
    builder_prompt = read_prompt("Builder", args.builder_prompt, args.builder_prompt_file)
    reviewer_prompt = read_prompt("Reviewer", args.reviewer_prompt, args.reviewer_prompt_file)
    if args.auto_protocol:
        builder_prompt = add_marker_instruction(builder_prompt, args.builder_marker, "Builder")
        reviewer_prompt = add_marker_instruction(reviewer_prompt, args.reviewer_marker, "Reviewer")
    builder = AgentSession(
        "Builder",
        builder_prompt,
        args.codex_cli,
        shlex.split(args.builder_args),
        log_dir,
    )
    reviewer = AgentSession(
        "Reviewer",
        reviewer_prompt,
        args.codex_cli,
        shlex.split(args.reviewer_args),
        log_dir,
    )
    await asyncio.gather(builder.start(), reviewer.start())
    try:
        if args.auto_protocol:
            coordinator = ProtocolCoordinator(
                builder=builder,
                reviewer=reviewer,
                objective=args.objective or "",
                reviewer_marker=args.reviewer_marker,
                builder_marker=args.builder_marker,
                max_turns=args.max_turns,
                turn_timeout=args.turn_timeout,
            )
            await coordinator.run()
        else:
            await interactive_loop(builder, reviewer)
    finally:
        await asyncio.gather(builder.stop(), reviewer.stop())


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.auto_protocol and not args.objective:
        parser.error("--objective is required when --auto-protocol is enabled.")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nSupervisor interrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
