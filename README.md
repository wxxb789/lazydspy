# lazydspy

`lazydspy` is a CLI/Agent that interviews you, understands your prompt-optimization scenario, and **generates a ready-to-run single-file DSPy optimization script**. The generated script follows modern Python practices (PEP 723 metadata, pydantic v2 models, Typer CLI, Rich output) and supports **quick/full** modes plus **checkpointing** for long runs. By design, `lazydspy` itself does not run the full optimization; it creates the script and guidance so you can run it with your own data and keys.

## Key ideas
- **Agent-first**: Uses Claude Agent SDK (v0.1.17) to ask concise, cost-aware questions and confirm your requirements.
- **Script generation**: Emits a single Python file with PEP 723 metadata declaring dependencies and `python >=3.12`.
- **Modes**: Quick mode samples a small subset for low-cost e2e checks; Full mode runs your full dataset with periodic checkpoints (defaults aim for ~10–20 checkpoints).
- **Data-first**: Default format is UTF-8 JSONL with validated input/output fields; a data guide and optional sample stubs are generated alongside the script.

## Setup
1. Ensure Python **3.12+** is available.
2. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```
3. Provide your API keys (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) via environment variables or an `.env` file. You can also override the system prompt with `LAZYDSPY_SYSTEM_PROMPT`.

## Usage
- Start an interactive session to generate a script (Agent-led Q&A):
  ```bash
  uv run lazydspy chat
  ```
  This produces a folder under `generated/<session_id>/` containing:
  - `pipeline.py` (PEP 723, Typer CLI, pydantic validation, checkpoints)
  - `metadata.json` (captured config), `DATA_GUIDE.md`, `README.md`
  - optional `sample-data/train.jsonl` stub if requested

- Run the generated script (examples):
  ```bash
  uv run generated/<session_id>/pipeline.py --mode quick
  uv run generated/<session_id>/pipeline.py --mode full --checkpoint-dir checkpoints --resume
  ```

## Custom Claude endpoint
`lazydspy` reads Claude Agent SDK settings from environment variables or CLI flags when starting the `chat` command. For a local Claude Code proxy (or other self-hosted endpoint), set:

```bash
# .env or shell
ANTHROPIC_BASE_URL=http://localhost:8030
ANTHROPIC_AUTH_TOKEN=dev-local-token
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Or via CLI
uv run lazydspy chat --base-url http://localhost:8030 --auth-token dev-local-token --model claude-3-5-sonnet-20241022
```

The endpoint must expose a Claude-compatible `/v1/messages` interface. Some proxies may not fully support streaming or advanced tool-calling; if initialization fails, `lazydspy` will fall back to local prompts with a warning.

- To layer on additional Claude Code controls (permissions + env), provide a JSON profile via `--agent-config`:
  ```json
  {
    "env": {
      "ANTHROPIC_BASE_URL": "http://localhost:4141",
      "ANTHROPIC_AUTH_TOKEN": "dummy",
      "ANTHROPIC_MODEL": "claude-opus-4.5",
      "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-opus-4.5",
      "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4.5",
      "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
      "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    },
    "permissions": {
      "deny": ["WebSearch"]
    }
  }
  ```
  Pass it to the CLI: `uv run lazydspy chat --agent-config ./agent.json --model claude-opus-4.5`.
  A starter `agent.json` with the above structure lives at the repo root—copy or modify it to suit your proxy settings.

## Checks
Run from repo root (recommended order):
1. `uv run ruff check .`
2. `uv run mypy`
3. `uv run pytest`

You can also execute them sequentially via `make check`.
