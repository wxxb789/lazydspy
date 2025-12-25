# lazydspy

`lazydspy` is an Agent-driven CLI tool that interviews you, understands your prompt-optimization scenario, and **generates a ready-to-run single-file DSPy optimization script**. The generated script follows modern Python practices (PEP 723 metadata, Pydantic v2 models, Typer CLI, Rich output) and supports **quick/full** modes plus **checkpointing** for long runs.

By design, `lazydspy` itself does not run the full optimization — it creates the script and guidance so you can run it with your own data and API keys.

## Key Features

- **Agent-first**: Uses Claude Agent SDK to dynamically collect requirements through natural conversation, not hardcoded question lists
- **Dynamic script generation**: The Agent writes code based on your specific needs, not template filling
- **PEP 723 compliant**: Generated scripts include inline metadata for dependency management
- **Modern Python stack**: Pydantic v2 for validation, Typer CLI, Rich terminal output
- **Dual run modes**: Quick mode for low-cost exploration, Full mode for production optimization
- **Checkpoint support**: Periodic checkpoints for long-running optimizations with resume capability

## Quick Start

### Prerequisites

- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key (Claude)

### Installation

```bash
git clone https://github.com/your-org/lazydspy.git
cd lazydspy
uv sync
```

### Basic Usage

Start an interactive conversation to generate a script:

```bash
uv run lazydspy chat
```

Or run directly (defaults to `chat`):

```bash
uv run lazydspy
```

## CLI Options

```bash
uv run lazydspy chat [OPTIONS]

Options:
  -m, --model TEXT       Claude model name (default: claude-opus-4.5)
  --base-url TEXT        Custom API endpoint
  --auth-token TEXT      API token (or set ANTHROPIC_API_KEY env var)
  --debug                Enable debug mode
  -v, --version          Show version
  --help                 Show help message
```

## Generated Output

After the conversation, `lazydspy` creates a folder under `generated/<session_id>/`:

```
generated/<session_id>/
├── pipeline.py      # Main optimization script (PEP 723 compliant)
├── metadata.json    # Configuration used
├── README.md        # Usage instructions
├── DATA_GUIDE.md    # Data preparation guide
└── sample-data/     # Optional sample JSONL
    └── train.jsonl
```

### Running the Generated Script

```bash
# Quick mode - low-cost exploration
uv run generated/<session_id>/pipeline.py --mode quick

# Full mode with checkpointing
uv run generated/<session_id>/pipeline.py --mode full --checkpoint-dir checkpoints --resume
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `ANTHROPIC_MODEL` | Model name (default: `claude-opus-4.5`) |
| `ANTHROPIC_BASE_URL` | Custom API endpoint (optional) |
| `ANTHROPIC_AUTH_TOKEN` | Alternative to ANTHROPIC_API_KEY |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI models) |
| `LAZYDSPY_DEBUG` | Enable debug mode (`1`, `true`, `yes`) |

### Custom Claude Endpoint

For local Claude proxy or self-hosted endpoints:

```bash
# Via environment variables
export ANTHROPIC_BASE_URL=http://localhost:8030
export ANTHROPIC_AUTH_TOKEN=dev-local-token
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Or via CLI
uv run lazydspy chat --base-url http://localhost:8030 --auth-token dev-local-token --model claude-3-5-sonnet-20241022
```

## Optimizers

### GEPA (GEvalPromptedAssembly)

Low-cost evolutionary prompt optimizer, good for general tasks.

**Best for**: Summarization, classification, generation, general tasks

| Mode | breadth | depth | temperature |
|------|---------|-------|-------------|
| quick | 2 | 2 | 0.3 |
| full | 4 | 4 | 0.7 |

### MIPROv2 (Model-based Instruction Prompt Refinement Optimizer v2)

Model-based instruction optimizer, better for complex reasoning.

**Best for**: Retrieval, scoring, QA, reasoning

| Mode | search_size | temperature |
|------|-------------|-------------|
| quick | 8 | 0.3 |
| full | 16 | 0.6 |

### Cost Comparison

- Quick mode is typically 5-10x cheaper than Full mode
- **Recommendation**: Start with `quick` to validate, then run `full` for production

## Architecture

The project follows an **Agentic Architecture**:

1. **Agent driven by System Prompt** (`agent/prompts.py`), not hardcoded logic
2. **Tools via MCP pattern** (`tools/`) for file, data, and domain operations
3. **Dynamic script generation** — Agent writes code, not template filling

```
src/lazydspy/
├── agent/           # Core Agent module
│   ├── config.py    # AgentConfig
│   ├── prompts.py   # SYSTEM_PROMPT
│   ├── runner.py    # AgentRunner
│   └── session.py   # ConversationSession
│
├── models/          # Pydantic data models
│   ├── config.py    # GenerationConfig
│   └── hyperparams.py
│
├── knowledge/       # Domain knowledge
│   ├── optimizers.py
│   └── cost_models.py
│
└── tools/           # MCP Tools
    ├── file_ops.py
    ├── data_ops.py
    └── domain_ops.py
```

## Development

### Code Checks

Run from repo root (recommended order):

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Or run all sequentially:

```bash
make check
```

### Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_cli_session.py::test_generation_config_overrides_are_typed -v
```

## Dependencies

**Core**:
- `anthropic` - Claude API client
- `dspy` - DSPy framework
- `pydantic>=2` - Data validation
- `typer>=0.12` - CLI framework
- `rich` - Terminal formatting
- `claude-agent-sdk==0.1.17` - Claude Agent SDK

**Dev**:
- `pytest>=8.3` - Testing
- `mypy>=1.11` - Type checking
- `ruff>=0.6` - Linting

## Troubleshooting

### Chinese Output Issues on Windows

Use PowerShell instead of cmd:

```bash
pwsh -NoProfile -Command "uv run lazydspy chat"
```

### API Token Not Set

Provide via environment variable or CLI:

```bash
export ANTHROPIC_API_KEY=your-key
# or
uv run lazydspy chat --auth-token your-key
```

## License

MIT
