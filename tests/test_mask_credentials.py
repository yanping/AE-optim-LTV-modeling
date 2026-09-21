"""
Unit tests for credentials masking and sanitization utility.
"""

from pathlib import Path
import re
import pytest

from scripts.mask_credentials import (
    extract_current_credentials,
    mask_files,
    restore_files,
    save_credentials_backup,
    load_credentials_backup,
    PROJECT_ID_PLACEHOLDER,
    GE_APP_ID_PLACEHOLDER,
    PROJECT_NUM_PLACEHOLDER,
)


def test_script_itself_has_no_hardcoded_secrets():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "mask_credentials.py"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")

    # Verify no sensitive keywords or project names are hardcoded
    assert "spartan-figure" not in content
    assert "1784542814283" not in content
    assert "20032108320" not in content


def test_extract_current_credentials(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
gcp:
  project_id: "test-proj-123"
  ge_app_id: "test-app-456"
""", encoding="utf-8")

    pid, app = extract_current_credentials(cfg_file)
    assert pid == "test-proj-123"
    assert app == "test-app-456"


def test_extract_placeholder_credentials(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"""
gcp:
  project_id: "{PROJECT_ID_PLACEHOLDER}"
  ge_app_id: "{GE_APP_ID_PLACEHOLDER}"
""", encoding="utf-8")

    pid, app = extract_current_credentials(cfg_file)
    assert pid is None
    assert app is None


def test_mask_files_and_dry_run(tmp_path):
    # Setup dummy repo structure
    cfg = tmp_path / "config.yaml"
    cfg.write_text('gcp:\n  project_id: "secret-proj"\n  ge_app_id: "secret-app"\n', encoding="utf-8")

    doc = tmp_path / "README.md"
    doc.write_text('Run gcloud --project=secret-proj with secret-app\n', encoding="utf-8")

    hist = tmp_path / "history.json"
    hist.write_text('{"name": "projects/9988776655/locations/global/collections/default_collection/engines/secret-app"}\n', encoding="utf-8")

    # 1. Dry run
    mods_dry = mask_files(tmp_path, "secret-proj", "secret-app", dry_run=True)
    assert len(mods_dry) == 3
    # Check files were NOT modified in dry-run
    assert "secret-proj" in cfg.read_text(encoding="utf-8")

    # 2. Real run
    mods = mask_files(tmp_path, "secret-proj", "secret-app", dry_run=False)
    assert len(mods) == 3

    cfg_content = cfg.read_text(encoding="utf-8")
    assert PROJECT_ID_PLACEHOLDER in cfg_content
    assert GE_APP_ID_PLACEHOLDER in cfg_content
    assert "请在此填入您的 Google Cloud Project ID" in cfg_content
    assert "secret-proj" not in cfg_content
    assert "secret-app" not in cfg_content

    doc_content = doc.read_text(encoding="utf-8")
    assert PROJECT_ID_PLACEHOLDER in doc_content
    assert "secret-proj" not in doc_content

    hist_content = hist.read_text(encoding="utf-8")
    assert PROJECT_NUM_PLACEHOLDER in hist_content
    assert "9988776655" not in hist_content


def test_restore_files(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text(f'Run gcloud --project={PROJECT_ID_PLACEHOLDER} with {GE_APP_ID_PLACEHOLDER}\n', encoding="utf-8")

    mods = restore_files(tmp_path, "client-proj", "client-app", dry_run=False)
    assert len(mods) == 1

    content = doc.read_text(encoding="utf-8")
    assert "client-proj" in content
    assert "client-app" in content
    assert PROJECT_ID_PLACEHOLDER not in content
    assert GE_APP_ID_PLACEHOLDER not in content


def test_backup_and_load_credentials(tmp_path):
    # Test saving backup
    backup_file = save_credentials_backup(tmp_path, "backup-proj-123", "backup-app-456")
    assert backup_file.exists()
    assert backup_file.name == ".credentials.backup"

    # Test loading from backup
    pid, app = load_credentials_backup(tmp_path)
    assert pid == "backup-proj-123"
    assert app == "backup-app-456"

    # Test loading from non-existent backup
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    none_pid, none_app = load_credentials_backup(empty_dir)
    assert none_pid is None
    assert none_app is None
