# Exact command and environment

Fixed command: `uv run --frozen python -m reproduction.run`

Environment: repository `uv.lock`, Python 3.12, one repository `.venv`.
Formal execution: Hugging Face `cpu-upgrade`, image
`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. The runner prints the exact
Git SHA, logical/affinity CPU allocation, estimate, and runtime.
