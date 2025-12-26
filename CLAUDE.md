# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Response Language

Always respond in Chinese, regardless of what language the user communicates in.

## Shell Commands

Use PowerShell (`pwsh -NoProfile`) instead of `cmd` when running commands that may output Chinese characters. This ensures proper Unicode/UTF-8 encoding support on Windows. Always use `-NoProfile` to skip loading user profile scripts and avoid unnecessary warnings.

Example:
```bash
pwsh -NoProfile -Command "uv run lazydspy --help"
```

## Project Overview

**lazydspy** is an Agent-driven CLI tool that helps users generate DSPy prompt optimization scripts through interactive dialogue. Built on **Claude Agent SDK**, it uses preset `claude_code` capabilities extended with custom MCP tools.

### Key Concepts

- **DSPy**: A framework for algorithmically optimizing LM prompts and weights
- **GEPA (GEvalPromptedAssembly)**: Low-cost evolutionary prompt optimizer, good for general tasks
- **MIPROv2**: Model-based instruction optimizer, better for complex reasoning tasks
- **Run Modes**: `quick` (low-cost exploration) vs `full` (production optimization)

## Architecture

The project uses **Claude Agent SDK** with a streamlined architecture:
1. Uses SDK's built-in tools (Read, Write, Edit, Bash, Glob, Grep)
2. Custom MCP tools for domain-specific operations (@tool decorator)
3. System prompt extends `claude_code` preset via `append` mode
4. Multi-turn conversation managed by ClaudeSDKClient

### Directory Structure

```
src/lazydspy/
├── __init__.py       # Package exports, version 0.2.0
├── __main__.py       # CLI entry point
├── cli.py            # Typer CLI (~120 lines)
├── agent.py          # Agent core with ClaudeSDKClient (~160 lines)
├── tools.py          # MCP tools with @tool decorator (~160 lines)
├── prompts.py        # System prompt config (~80 lines)
└── knowledge/        # Domain knowledge (unchanged)
    ├── __init__.py
    ├── optimizers.py # OptimizerInfo, OPTIMIZER_REGISTRY
    └── cost_models.py # MODEL_PRICING, estimate_optimization_cost
```

## Common Commands

### Development

```bash
# Install dependencies
uv sync

# Run the CLI
uv run lazydspy
uv run lazydspy chat
uv run lazydspy --help

# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/lazydspy/

# Linting
uv run ruff check src/
uv run ruff format src/
```

### Testing Notes

- Tests are in `tests/` directory
- `conftest.py` installs stubs for `rich`, `typer`, and `claude_agent_sdk`
- Use `run_async()` helper for testing async tool functions
- Tool implementation functions (e.g. `estimate_cost_impl`) are exported separately for testing

## Key Files

### Agent (`agent.py`)

Core Agent using Claude Agent SDK:
- `AgentConfig`: Configuration dataclass (model, debug, workdir)
- `Agent`: Main class with `run()` method for conversation loop
- Uses `ClaudeSDKClient` for multi-turn conversations

### Tools (`tools.py`)

MCP tools using @tool decorator:
- `estimate_cost`: Estimate optimization costs
- `list_optimizers`: List available optimizers
- `get_defaults`: Get default hyperparameters
- Business logic in `*_impl` functions for easier testing

### Prompts (`prompts.py`)

System prompt configuration:
- Uses `preset: claude_code` with `append` mode
- `SYSTEM_PROMPT_APPEND` contains lazydspy-specific instructions
- `get_system_prompt_config()` returns the configuration dict

### Knowledge (`knowledge/`)

Domain knowledge (unchanged from v0.1.0):
- `optimizers.py`: Optimizer registry and recommendations
- `cost_models.py`: Cost estimation for different models

## Code Style

- Python 3.12+
- Type hints throughout
- Ruff for linting and formatting (line length 100)
- Chinese comments in domain-specific code
- English for docstrings and public API

## Dependencies

Core:
- `claude-agent-sdk>=0.1.17` - Claude Agent SDK
- `rich` - Terminal formatting
- `typer>=0.12` - CLI framework

Dev:
- `pytest>=8.3` - Testing
- `mypy>=1.11` - Type checking
- `ruff>=0.6` - Linting

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_AUTH_TOKEN` | Claude API token (priority 1) |
| `ANTHROPIC_API_KEY` | Claude API key (priority 2) |
| `ANTHROPIC_MODEL` | Model name (default: `claude-sonnet-4-20250514`) |
| `ANTHROPIC_BASE_URL` | Custom API endpoint (optional) |
| `LAZYDSPY_DEBUG` | Enable debug mode (`1`, `true`, `yes`) |

## Generated Output Structure

When the Agent generates a script, it creates:

```
generated/<session_id>/
├── pipeline.py      # Main optimization script (PEP 723 compliant)
├── metadata.json    # Configuration used
└── README.md        # Usage instructions
```

## Optimizer Presets

### GEPA
| Mode | breadth | depth | temperature |
|------|---------|-------|-------------|
| quick | 2 | 2 | 0.3 |
| full | 4 | 4 | 0.7 |

### MIPROv2
| Mode | search_size | temperature |
|------|-------------|-------------|
| quick | 8 | 0.3 |
| full | 16 | 0.6 |

## Troubleshooting

### Import Errors
If tests fail with import errors, check:
1. `sys.path` order - ensure real packages are found before stubs
2. `conftest.py` properly installs stubs before imports

### Chinese Output Issues on Windows
Use PowerShell with `-NoProfile` flag instead of cmd:
```bash
pwsh -NoProfile -Command "uv run lazydspy chat"
```

### API Key Not Found
Set environment variable:
```bash
export ANTHROPIC_API_KEY=your-key
# or
export ANTHROPIC_AUTH_TOKEN=your-token
```
