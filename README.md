# lazydspy

## Setup

1. Ensure Python 3.12+ is available.
2. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and provide your secrets (e.g., `OPENAI_API_KEY`).

The codebase uses a `src/` layout, with linting configured through `ruff` and type checking through `mypy` in strict mode.
