from pathlib import Path

from backend.runtime import ensure_runtime_tree


def test_ensure_runtime_tree_preserves_existing_database(tmp_path: Path) -> None:
    root = tmp_path / "ETF-Investment-Lab"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    database = data_dir / "investment_lab.db"
    original_contents = b"existing-local-investment-data"
    database.write_bytes(original_contents)

    ensure_runtime_tree(root)

    assert database.read_bytes() == original_contents
    for runtime_directory in (
        data_dir,
        data_dir / "backups",
        root / "config",
        root / "reports",
        root / "reports" / "exports",
    ):
        assert runtime_directory.is_dir()
