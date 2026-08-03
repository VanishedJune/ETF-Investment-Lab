"""Project-relative runtime directory helpers."""

from pathlib import Path


RUNTIME_DIRECTORIES = (
    Path("data"),
    Path("data") / "backups",
    Path("config"),
    Path("reports"),
    Path("reports") / "exports",
)


def ensure_runtime_tree(root: Path | str) -> Path:
    """Create required local directories without altering existing files."""
    project_root = Path(root).resolve()
    for relative_directory in RUNTIME_DIRECTORIES:
        (project_root / relative_directory).mkdir(parents=True, exist_ok=True)
    return project_root
