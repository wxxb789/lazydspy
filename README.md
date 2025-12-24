# lazydspy

## Setup

1. Ensure Python 3.12+ is available.
2. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and provide your secrets (e.g., `OPENAI_API_KEY`).

Key environment variables:

- `ANTHROPIC_API_KEY`: Claude Agent SDK 将使用的 API Key（或兼容变量，如 `CLAUDE_API_KEY`）。  
- `LAZYDSPY_SYSTEM_PROMPT`: 自定义交互式问答的系统提示。

The codebase uses a `src/` layout, with linting configured through `ruff` and type checking through `mypy` in strict mode.

## CLI Usage

The outer CLI is now powered by [Typer](https://typer.tiangolo.com/) with Rich output:

- Start an interactive Claude-guided chat to collect pipeline config:

  ```bash
  uv run lazydspy chat
  ```

- Generated `pipeline.py` scripts embed dependencies via [PEP 723](https://peps.python.org/pep-0723/) blocks, so you can run them directly with uv, e.g.:

  ```bash
  uv run pipeline.py --mode quick
  ```

- Optimization should be performed with the generated script (see the `uv run pipeline.py` example above); the previous bundled demo has been removed.

## Checks

Run the following commands from the repository root after setup (recommended order):

1. `ruff check .`
2. `mypy`
3. `pytest`

You can also execute them sequentially via `make check`.
