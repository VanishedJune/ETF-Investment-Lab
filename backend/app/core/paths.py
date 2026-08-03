"""Project and packaged-resource locations used by the local application."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the writable application home (beside the EXE when packaged)."""
    configured = os.environ.get("INVESTMENT_LAB_HOME")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3]


def resource_root() -> Path:
    """Return the read-only source/resource tree bundled with the executable."""
    configured = os.environ.get("INVESTMENT_LAB_RESOURCE_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3]


def data_directory() -> Path:
    """Return the local data directory without creating or changing files."""
    return project_root() / "data"


def database_path() -> Path:
    """Return the required project-relative SQLite database location."""
    return data_directory() / "investment_lab.db"


def config_directory() -> Path:
    """Return the project-local configuration directory."""
    return project_root() / "config"
