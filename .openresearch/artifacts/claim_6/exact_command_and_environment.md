# Exact command and environment

Fixed command: `uv run --frozen python -m reproduction.run`

Environment: repository `uv.lock`, Python 3.12, one repository `.venv`.
Formal execution: Hugging Face `cpu-upgrade`, image
`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. Estimated peak use is 54
cores: nine baseline workers at six PyTorch threads; the earlier EBiEOT phase
uses six workers at eight threads. Actual allocation and runtime are printed.
