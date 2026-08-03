"""Project-local backup, restore, and portable ZIP export operations."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from ..core.paths import project_root


BACKUP_FOLDERS = (
    "config",
    "reports",
    "data/imports",
    "data/model_iterations",
    "data/weekly_analysis_v2",
)
BACKUP_DATABASE = "data/investment_lab.db"
BACKUP_IGNORE = shutil.ignore_patterns(
    "pytest-*",
    ".pytest-*",
    "test-results",
    "playwright-report",
    "__pycache__",
)


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create an atomic, WAL-aware SQLite snapshot.

    Copying only the main ``.db`` file can lose committed WAL transactions and
    can capture a database in the middle of a checkpoint.  SQLite's online
    backup API provides one transactionally consistent image even while the
    application is open.  The destination is replaced only after ``quick_check``
    succeeds.
    """

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_uri = f"{source.as_uri()}?mode=ro"
        with closing(
            sqlite3.connect(source_uri, uri=True, timeout=30)
        ) as source_connection, closing(
            sqlite3.connect(temporary, timeout=30)
        ) as destination_connection:
            source_connection.backup(destination_connection)
            result = destination_connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise sqlite3.DatabaseError(
                    f"SQLite snapshot failed quick_check: {result!r}"
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_sqlite_database(source: Path, destination: Path) -> None:
    """Restore through SQLite so an open Windows database is not replaced.

    The web application keeps pooled SQLite handles open.  Windows therefore
    rejects ``os.replace`` on the live database.  The online backup API writes
    through SQLite's locking and transaction machinery, so existing pooled
    connections observe the restored image safely.
    """

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source.as_uri()}?mode=ro"
    with closing(
        sqlite3.connect(source_uri, uri=True, timeout=30)
    ) as source_connection, closing(
        sqlite3.connect(destination, timeout=30)
    ) as destination_connection:
        source_connection.backup(destination_connection)
        result = destination_connection.execute("PRAGMA quick_check").fetchone()
        if result != ("ok",):
            raise sqlite3.DatabaseError(
                f"Restored SQLite database failed quick_check: {result!r}"
            )


class BackupService:
    """Copies only project-owned data; application source code is never modified."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or project_root()).resolve()
        self.backups_root = self.root / "data" / "backups"
        self.exports_root = self.root / "data" / "exports"
        self.backups_root.mkdir(parents=True, exist_ok=True)
        self.exports_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stamp() -> str:
        # Microseconds avoid collisions when a restore creates its automatic
        # safety backup immediately after a user-created backup.
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    def _copy_payload(self, target: Path) -> None:
        database = self.root / BACKUP_DATABASE
        if database.exists():
            destination = target / BACKUP_DATABASE
            snapshot_sqlite_database(database, destination)
        for relative in BACKUP_FOLDERS:
            source = self.root / relative
            if source.exists():
                shutil.copytree(
                    source,
                    target / relative,
                    dirs_exist_ok=True,
                    ignore=BACKUP_IGNORE,
                )
        (target / "manifest.json").write_text(
            json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "format": 2, "contents": [BACKUP_DATABASE, *BACKUP_FOLDERS]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(self) -> dict[str, object]:
        folder = self.backups_root / self._stamp()
        folder.mkdir(parents=True, exist_ok=False)
        try:
            self._copy_payload(folder)
        except Exception:
            # A failed copy is not a restorable backup and must not remain in
            # the list as a misleading partial artifact.
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return self.describe(folder)

    def describe(self, folder: Path) -> dict[str, object]:
        return {
            "name": folder.name,
            "created_at": datetime.fromtimestamp(folder.stat().st_mtime).isoformat(timespec="seconds"),
            "has_database": (folder / BACKUP_DATABASE).exists(),
            "path": str(folder.relative_to(self.root)),
        }

    def list(self) -> list[dict[str, object]]:
        return [self.describe(folder) for folder in sorted(self.backups_root.iterdir(), reverse=True) if folder.is_dir() and (folder / "manifest.json").exists()]

    def _backup(self, name: str) -> Path:
        candidate = (self.backups_root / name).resolve()
        if candidate.parent != self.backups_root.resolve() or not (candidate / "manifest.json").exists():
            raise ValueError("Unknown backup")
        return candidate

    def restore(self, name: str) -> dict[str, object]:
        source = self._backup(name)
        # Preserve before restoring, so a mistaken restore can itself be undone.
        safety = self.create()
        for relative in (BACKUP_DATABASE, *BACKUP_FOLDERS):
            saved = source / relative
            destination = self.root / relative
            if saved.is_file():
                if relative == BACKUP_DATABASE:
                    restore_sqlite_database(saved, destination)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    shutil.copy2(saved, temporary_path)
                    temporary_path.replace(destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
            elif saved.is_dir():
                shutil.copytree(saved, destination, dirs_exist_ok=True)
        return {"restored": name, "safety_backup": safety["name"]}

    def export_zip(self) -> Path:
        export_path = self.exports_root / f"investment-lab-export-{self._stamp()}.zip"
        with tempfile.TemporaryDirectory(dir=self.exports_root) as staging_text:
            staging = Path(staging_text)
            self._copy_payload(staging)
            with ZipFile(export_path, "w", ZIP_DEFLATED) as archive:
                for file in staging.rglob("*"):
                    if file.is_file():
                        archive.write(file, file.relative_to(staging).as_posix())
        return export_path

    def import_zip(self, content: bytes) -> dict[str, object]:
        if not content:
            raise ValueError("The import archive is empty")
        with tempfile.TemporaryDirectory(dir=self.exports_root) as staging_text:
            staging = Path(staging_text)
            archive_path = staging / "import.zip"
            archive_path.write_bytes(content)
            with ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    candidate = (staging / member.filename).resolve()
                    if candidate != staging and staging not in candidate.parents:
                        raise ValueError("Archive contains an unsafe path")
                archive.extractall(staging / "payload")
            payload = staging / "payload"
            if not (payload / "manifest.json").exists():
                raise ValueError("Archive is not an ETF Investment Lab data export")
            imported = self.backups_root / f"imported-{self._stamp()}"
            shutil.copytree(payload, imported)
        return self.restore(imported.name)
