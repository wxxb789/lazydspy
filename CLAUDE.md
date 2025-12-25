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

**lazydspy** is an Agent-driven CLI tool that helps users generate DSPy prompt optimization scripts through interactive dialogue. Instead of hardcoded question lists and templates, the Agent dynamically collects requirements and generates complete, ready-to-run Python scripts.

### Key Concepts

- **DSPy**: A framework for algorithmically optimizing LM prompts and weights
- **GEPA (GEvalPromptedAssembly)**: Low-cost evolutionary prompt optimizer, good for general tasks
- **MIPROv2**: Model-based instruction optimizer, better for complex reasoning tasks
- **Run Modes**: `quick` (low-cost exploration) vs `full` (production optimization)

## Architecture

The project follows an **Agentic Architecture** where:
1. The Agent is driven by a system prompt (`agent/prompts.py`), not hardcoded logic
2. Tools are provided via MCP (Model Context Protocol) pattern
3. Scripts are generated dynamically by the Agent, not from templates

### Directory Structure

```
src/lazydspy/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── cli.py               # Typer CLI (thin wrapper around AgentRunner)
├── schemas.py           # MetricResult, ScoreDetail schemas
│
├── agent/               # Core Agent module
│   ├── config.py        # AgentConfig - model, auth, settings
│   ├── prompts.py       # SYSTEM_PROMPT - Agent behavior definition
│   ├── runner.py        # AgentRunner - conversation loop + tool execution
│   └── session.py       # ConversationSession - message history
│
├── models/              # Pydantic data models
│   ├── config.py        # GenerationConfig - unified generation settings
│   └── hyperparams.py   # GEPAHyperparameters, MIPROv2Hyperparameters, presets
│
├── knowledge/           # Domain knowledge
│   ├── optimizers.py    # OptimizerInfo, OPTIMIZER_REGISTRY
│   └── cost_models.py   # MODEL_PRICING, estimate_optimization_cost
│
└── tools/               # MCP Tools for Agent
    ├── file_ops.py      # write_file, read_file, create_dir
    ├── data_ops.py      # validate_jsonl, check_schema, sample_data
    └── domain_ops.py    # estimate_cost, list_optimizers, get_defaults

src/                     # Legacy/stub modules (for tests)
├── core.py              # WebSummarySignature, DeepSummarizer (example DSPy module)
├── dspy.py              # Lightweight dspy stub for local testing
├── metrics.py           # llm_judge_metric using OpenAI
└── pydantic.py          # Pydantic stub for environments without real pydantic
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

# Run specific test
uv run pytest tests/test_cli_session.py::test_generation_config_overrides_are_typed -v

# Type checking
uv run mypy src/lazydspy/

# Linting
uv run ruff check src/
uv run ruff format src/
```

### Testing Notes

- Tests are in `tests/` directory
- `conftest.py` installs stubs for `rich`, `anthropic`, and `typer` to allow tests to run without network dependencies
- Use `run_async()` helper for testing async tool functions (not `@pytest.mark.asyncio`)
- The `src/pydantic.py` stub exists for legacy compatibility - ensure tests use real pydantic by appending (not inserting) src path

## Key Files

### Agent System Prompt (`agent/prompts.py`)

This is the core of the Agentic architecture. The SYSTEM_PROMPT defines:
- Agent role and behavior guidelines
- What information to collect (scenario, fields, model, optimizer, mode)
- Script generation standards (PEP 723, Typer CLI, Pydantic v2)
- Available tools and when to use them
- Output directory structure

### GenerationConfig (`models/config.py`)

Unified configuration for script generation with:
- Field validators for algorithm, mode, fields parsing
- Hyperparameter preset application via `model_validator`
- Support for both GEPA and MIPROv2 hyperparameters

**Important**: The `hyperparameters` field uses `Any` type to avoid Pydantic v2's Union type coercion, which would convert dict to model instance before validators run.

### Tools (`tools/`)

All tools return a standardized format:
```python
{
    "content": [
        {"type": "text", "text": "..."}
    ]
}
```

Tools are registered in `tools/__init__.py` and exposed via `get_all_tool_schemas()`.

## Code Style

- Python 3.12+
- Pydantic v2 for data validation
- Type hints throughout (strict mypy)
- Ruff for linting and formatting (line length 100)
- Chinese comments in domain-specific code
- English for docstrings and public API

## Dependencies

Core:
- `dspy` - DSPy framework
- `pydantic>=2` - Data validation
- `typer>=0.12` - CLI framework
- `rich` - Terminal formatting
- `claude-agent-sdk==0.1.17` - Claude Agent SDK

Dev:
- `pytest>=8.3` - Testing
- `mypy>=1.11` - Type checking
- `ruff>=0.6` - Linting

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key (required) |
| `ANTHROPIC_MODEL` | Model name (default: `claude-opus-4.5`) |
| `ANTHROPIC_BASE_URL` | Custom API endpoint (optional) |
| `LAZYDSPY_DEBUG` | Enable debug mode (`1`, `true`, `yes`) |

## Generated Output Structure

When the Agent generates a script, it creates:

```
generated/<session_id>/
├── pipeline.py      # Main optimization script (PEP 723 compliant)
├── metadata.json    # Configuration used
├── README.md        # Usage instructions
├── DATA_GUIDE.md    # Data preparation guide
└── sample-data/     # Optional sample JSONL
    └── train.jsonl
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

### Hyperparameter Preset Not Applied
The `GenerationConfig` uses a model validator to apply presets. If presets aren't applied:
1. Check that `hyperparameters` is passed as a dict, not a model instance
2. The validator applies presets first, then user overrides

### Chinese Output Issues on Windows
Use PowerShell with `-NoProfile` flag instead of cmd:
```bash
pwsh -NoProfile -Command "uv run lazydspy chat"
```
