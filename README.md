# Supervisor

Supervisor orchestrates two Codex CLI agents – a Builder and a Reviewer – so they can collaborate on the same repository. The Reviewer inspects repo state, writes structured prompts, and the Builder executes edits/tests. A strict protocol keeps turns synchronized, injects repo context, persists everything for auditing, and can even auto-commit when the Reviewer approves.

## Highlights
- **Structured protocol:** Reviewer must emit `SUMMARY`, `PROMPT`, `FILES`, `CONTEXT` blocks plus the `<<REVIEWER_DONE>>` marker. Builder responses are bounded and marker-terminated to prevent runaway output.
- **Rich context feeds:** Every reviewer turn includes `git status -sb`, `git diff --stat`, the previous builder report, and summaries of automatic tool runs. Builder prompts can include reviewer-selected file excerpts and extra context text.
- **Automatic tools & commits:** Configure repeated commands (tests, lint, etc.) with `--builder-tool`. Their output flows back into the next Reviewer/Builder turn. Optional `--auto-commit-each-turn` / `--auto-commit-final` hooks keep the repo up to date once work is reviewed.
- **Artifact logging & resilience:** Each session writes to `~/.codex-supervisor/logs/session-*/turns.jsonl`, `transcript.md`, optional git snapshots, and an incremental `state.json`. Combined with automatic Codex restarts and `--resume-session`, you can list, view, or continue any run without losing context.
- **Configurable everything:** Use CLI flags or a YAML/JSON config file to set repo paths, timeouts, context line limits, tool timeouts, commit templates, etc.
- **Manual REPL fallback:** Skip `--auto-protocol` to drive either agent interactively with `b:` / `r:` prompts.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```
PyYAML is required for YAML configs and is installed automatically via `pip install .`.

Or run the bundled helper script:
```bash
bash install.sh
```

### Optional: Build a macOS `.app`
1. Install PyInstaller inside the virtual env: `pip install pyinstaller`.
2. Convert the included icon (`assets/icon.png`) into `.icns`:
   ```bash
   mkdir -p CodexSupervisor.iconset
   sips -z 1024 1024 assets/icon.png --out CodexSupervisor.iconset/icon_512x512@2x.png
   sips -z 512 512 assets/icon.png --out CodexSupervisor.iconset/icon_512x512.png
   sips -z 256 256 assets/icon.png --out CodexSupervisor.iconset/icon_256x256.png
   sips -z 128 128 assets/icon.png --out CodexSupervisor.iconset/icon_128x128.png
   sips -z 64 64 assets/icon.png --out CodexSupervisor.iconset/icon_32x32@2x.png
   sips -z 32 32 assets/icon.png --out CodexSupervisor.iconset/icon_32x32.png
   sips -z 16 16 assets/icon.png --out CodexSupervisor.iconset/icon_16x16.png
   cp CodexSupervisor.iconset/icon_32x32.png CodexSupervisor.iconset/icon_16x16@2x.png
   iconutil -c icns CodexSupervisor.iconset
   ```
3. Build the bundle: `pyinstaller --windowed --icon CodexSupervisor.icns --name "Codex Supervisor" gui.py`
4. Launch/pin it: `open dist/Codex\ Supervisor.app` → right-click Dock icon → Keep in Dock.

## Usage
### macOS GUI Launcher
```bash
python3 gui.py
# or, after `pip install .`, simply run:
codex-supervisor-gui
```
The window lets you pick a config file or fill in the objective/repo path fields, then start/stop the supervisor with live logs. On macOS you can bundle it as an app via:
```bash
pip install pyinstaller
pyinstaller --windowed --name "Codex Supervisor" gui.py
open dist/Codex\ Supervisor.app
```
Make sure the `Codex CLI path` field points to your installed `codex` binary (defaults to `codex`, so it works if the CLI is on `PATH`).

### Manual REPL
```bash
codex-supervisor --codex-cli codex
```

### Structured Protocol with Config File
```yaml
# supervisor.yaml
objective: "Ship dark-mode settings page"
auto_protocol: true
repo_path: .
builder_tool:
  - "pytest -q"
  - "ruff check"
max_turns: 12
turn_timeout: 360
auto_commit_final: true
```
```bash
codex-supervisor --config supervisor.yaml
```

### Direct CLI Example
```bash
codex-supervisor \
  --auto-protocol \
  --objective "Refactor search pipeline" \
  --repo-path . \
  --builder-tool "pytest -q" \
  --builder-tool "ruff check" \
  --auto-commit-final \
  --max-turns 8
```

### Reviewer Format Reminder
```
SUMMARY: <brief recap>
PROMPT: <instructions or APPROVED>
FILES: path/to/file.py another/file.ts
CONTEXT: optional extra details
<<REVIEWER_DONE>>
```

### Builder Expectations
Builder should apply the instructions, run any needed repo commands, summarize the work, and finish with `<<BUILDER_DONE>>`. Output is capped with `--max-agent-output-lines` to prevent runaway transcripts.

## Session Management
- `--log-dir` now defaults to `~/.codex-supervisor/logs`. Each auto-protocol run creates `session-YYYYmmdd-HHMMSS/` folders there unless you pass a custom path (relative paths resolve against your current working directory).
- Every session folder contains `state.json`, which tracks the latest reviewer summary, builder report, and completed turn so `--resume-session` can pick up mid-stream.
- `codex-supervisor --list-sessions` lists known sessions.
- `codex-supervisor --show-session session-20240101-120000` prints the Markdown transcript.
- `codex-supervisor --auto-protocol ... --resume-session session-...` continues appending to an existing log directory.

## Git Snapshots & Commits
- Enable/disable per-turn snapshots with `--save-git-snapshots / --no-save-git-snapshots` (default: enabled). Files land next to the transcript.
- Auto commits:
  - `--auto-commit-each-turn` commits after every builder turn.
  - `--auto-commit-final` commits once the reviewer emits `PROMPT: APPROVED`.
  - `--commit-template "Supervisor turn {turn}: {summary}"` controls the commit message.

## Tests
```bash
python -m unittest discover tests
```
Config parsing tests cover YAML/JSON merging and protocol orchestration/resume logic via mocked agents.

## Tips
- Provide custom role prompts via `--builder-prompt-file` / `--reviewer-prompt-file` to tailor behaviour per project.
- Keep `FILES:` lists small; the supervisor truncates each excerpt to `--file-excerpt-lines` lines.
- Combine `--builder-tool` commands with `--tool-output-lines` to capture just the failure essentials while keeping transcripts compact.
