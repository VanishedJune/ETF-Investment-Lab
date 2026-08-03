from pathlib import Path
import sqlite3

from backend.app.services.backup_service import BackupService


def test_backup_keeps_user_reports_but_skips_test_artifacts(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    database = tmp_path / "data" / "investment_lab.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('sqlite-test')")
    (tmp_path / "reports" / "research").mkdir(parents=True)
    (tmp_path / "reports" / "research" / "result.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "reports" / "pytest-locked").mkdir()
    (tmp_path / "reports" / "pytest-locked" / "scratch.txt").write_text("skip", encoding="utf-8")

    result = BackupService(tmp_path).create()
    backup = tmp_path / str(result["path"])

    with sqlite3.connect(backup / "data" / "investment_lab.db") as connection:
        assert connection.execute("SELECT value FROM marker").fetchone() == (
            "sqlite-test",
        )
    assert (backup / "reports" / "research" / "result.txt").read_text("utf-8") == "keep"
    assert not (backup / "reports" / "pytest-locked").exists()


def test_backup_captures_committed_wal_rows_and_legacy_audit_folders(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    database = data / "investment_lab.db"
    live = sqlite3.connect(database)
    try:
        assert live.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        live.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
        live.execute("INSERT INTO marker VALUES (33)")
        live.commit()
        (data / "model_iterations" / "399006").mkdir(parents=True)
        (data / "model_iterations" / "399006" / "audit.json").write_text(
            "{}", encoding="utf-8"
        )
        (data / "weekly_analysis_v2").mkdir()
        (data / "weekly_analysis_v2" / "checkpoint.json").write_text(
            "{}", encoding="utf-8"
        )

        result = BackupService(tmp_path).create()
    finally:
        live.close()

    backup = tmp_path / str(result["path"])
    with sqlite3.connect(backup / "data" / "investment_lab.db") as connection:
        assert connection.execute("SELECT value FROM marker").fetchone() == (33,)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
    assert (backup / "data" / "model_iterations" / "399006" / "audit.json").is_file()
    assert (backup / "data" / "weekly_analysis_v2" / "checkpoint.json").is_file()


def test_restore_updates_database_while_an_existing_connection_is_open(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    database = data / "investment_lab.db"
    live = sqlite3.connect(database)
    try:
        live.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
        live.execute("INSERT INTO marker VALUES (1)")
        live.commit()
        service = BackupService(tmp_path)
        backup_name = str(service.create()["name"])

        live.execute("UPDATE marker SET value = 2")
        live.commit()
        result = service.restore(backup_name)

        assert result["restored"] == backup_name
        assert result["safety_backup"] != backup_name
        assert live.execute("SELECT value FROM marker").fetchone() == (1,)
        assert len(service.list()) == 2
    finally:
        live.close()
