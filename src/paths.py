"""Repository-root-relative path helpers (cross-platform via pathlib).

The repository root is resolved from this file's location, never from the current
working directory, so the pipeline runs identically on Windows and macOS/Linux and
regardless of where ``python`` is invoked.
"""

from __future__ import annotations

from pathlib import Path

# src/paths.py -> src/ -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    """Return an absolute path built from the repo root."""
    return REPO_ROOT.joinpath(*parts)
