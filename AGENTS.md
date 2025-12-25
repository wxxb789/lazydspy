# AGENTS Guidance for `lazydspy`

These rules apply to the entire repository.

## Product intent
- `lazydspy` itself is a **CLI/Agent** that **collects requirements and generates** a **single-file DSPy optimization script**; it should not directly run full optimization for the user.
- Generated scripts must:
  - Include **PEP 723 metadata** with Python `>=3.12` and required dependencies.
  - Offer **quick** and **full** modes; quick uses small subsets/low cost, full supports full data.
  - Implement **checkpointing** with ~10–20 checkpoints by default, `--checkpoint-interval`, `--max-checkpoints`, and `--resume`.
  - Use **JSONL** as the default data format and provide basic validation hints.
  - Print and persist the final optimized prompt.

## Tooling & dependencies
- CLI should be built with **Typer** + **Rich**.
- Use **Claude Agent SDK v0.1.17** for agent interactions (do not fall back to other SDKs unless explicitly required).
- Use **pydantic v2** for all config/state models; avoid untyped/dynamic `Dict[str, Any]` in new code when practical.
- Prefer **uv** for running checks; default commands: `uv run ruff check .`, `uv run mypy`, `uv run pytest`.

## Optimizer guidance
- Support and recommend **GEPA** and **MIPROv2**; start with the simplest, lowest-cost configuration and let users opt into heavier settings.
- Provide scenario-aware defaults (e.g., summarization vs. retrieval) and surface cost/complexity trade-offs in prompts or docs.

## Data guidance
- Default training/validation format: **UTF-8 JSONL** with explicit input/output fields confirmed via CLI.
- Provide users with data preparation tips or sample stubs when generating artifacts.

## Legacy notes
- The old `optimize` demo should not be treated as the primary workflow; prioritize the generation-first model.

## Process
- Keep instructions in this file up to date when workflows change.
- When touching files, re-read this file to ensure compliance.
