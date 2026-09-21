#!/usr/bin/env python3
"""
Credentials Sanitization & Masking Utility.
-----------------------------------------------------------------------------
Scans the project for personal GCP project_id and Gemini Enterprise ge_app_id,
replacing them with clear, standardized placeholder markers to prevent accidental
leakage of sensitive personal credentials when sharing the repository.

IMPORTANT:
This script contains NO hardcoded sensitive values. It dynamically detects the
current project_id and ge_app_id from config.yaml, or accepts them via CLI arguments.

Usage:
  # 1. Mask sensitive credentials (auto-detect from config.yaml):
  python3 scripts/mask_credentials.py

  # 2. Preview changes without modifying files (dry-run):
  python3 scripts/mask_credentials.py --dry-run

  # 3. Specify explicit credentials to mask:
  python3 scripts/mask_credentials.py --project-id my-project --app-id my-app

  # 4. Restore / inject new credentials into placeholder markers:
  python3 scripts/mask_credentials.py --restore --project-id new-project --app-id new-app
-----------------------------------------------------------------------------
"""

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ID_PLACEHOLDER = "<YOUR_GCP_PROJECT_ID>"
GE_APP_ID_PLACEHOLDER = "<YOUR_GE_APP_ID>"
PROJECT_NUM_PLACEHOLDER = "<YOUR_GCP_PROJECT_NUMBER>"
BACKUP_FILE_NAME = ".credentials.backup"

# Directories and files to exclude from scanning
EXCLUDE_DIRS = {
    "venv",
    ".venv",
    "env",
    "tests",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
}

# File extensions to scan
SCAN_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".md",
    ".py",
    ".json",
    ".sh",
    ".txt",
    ".html",
}


def find_repo_root() -> Path:
    """Finds repository root based on existence of config.yaml or README.md."""
    cur = Path(__file__).resolve().parent
    while cur != cur.parent:
        if (cur / "config.yaml").exists() and (cur / "README.md").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def extract_current_credentials(config_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Dynamically extracts the current project_id and ge_app_id from config.yaml
    without requiring third-party YAML libraries.
    """
    if not config_path.exists():
        return None, None

    content = config_path.read_text(encoding="utf-8")
    pid_match = re.search(r'^\s*project_id:\s*["\']?([^"\'#\s\n]+)["\']?', content, re.MULTILINE)
    app_match = re.search(r'^\s*ge_app_id:\s*["\']?([^"\'#\s\n]+)["\']?', content, re.MULTILINE)

    pid = pid_match.group(1).strip() if pid_match else None
    app = app_match.group(1).strip() if app_match else None

    # Check if they are already placeholders
    if pid and pid.startswith("<") and pid.endswith(">"):
        pid = None
    if app and app.startswith("<") and app.endswith(">"):
        app = None

    return pid, app


def save_credentials_backup(root_dir: Path, project_id: str, ge_app_id: str) -> Path:
    """Saves unmasked credentials into local .credentials.backup file."""
    backup_path = root_dir / BACKUP_FILE_NAME
    content = (
        f"# AlphaEvolve local credentials backup\n"
        f"# Generated automatically by scripts/mask_credentials.py\n"
        f"# DO NOT COMMIT THIS FILE (protected by .gitignore)\n"
        f"project_id: {project_id}\n"
        f"ge_app_id: {ge_app_id}\n"
    )
    backup_path.write_text(content, encoding="utf-8")
    return backup_path


def load_credentials_backup(root_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Loads backup credentials from local .credentials.backup if present."""
    backup_path = root_dir / BACKUP_FILE_NAME
    if not backup_path.exists():
        return None, None
    try:
        content = backup_path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    pid_match = re.search(r'^\s*project_id:\s*["\']?([^"\'#\s\n]+)["\']?', content, re.MULTILINE)
    app_match = re.search(r'^\s*ge_app_id:\s*["\']?([^"\'#\s\n]+)["\']?', content, re.MULTILINE)

    pid = pid_match.group(1).strip() if pid_match else None
    app = app_match.group(1).strip() if app_match else None
    return pid, app


def get_files_to_process(root_dir: Path) -> List[Path]:
    """Collects all relevant text and configuration files in the repo."""
    target_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SCAN_EXTENSIONS or fname in {"Makefile", "Dockerfile"}:
                fpath = Path(dirpath) / fname
                # Exclude the script itself
                if fpath.resolve() == Path(__file__).resolve():
                    continue
                target_files.append(fpath)
    return target_files


def mask_files(
    root_dir: Path,
    project_id: str,
    ge_app_id: str,
    dry_run: bool = False,
) -> Dict[Path, int]:
    """
    Replaces all occurrences of project_id and ge_app_id across files.
    Also anonymizes GCP project numbers in experiment paths.
    """
    files = get_files_to_process(root_dir)
    modifications = {}

    # Pattern for GCP experiment resource paths like: projects/123456789/locations/...
    proj_num_pattern = re.compile(r'projects/(\d{8,14})/locations/global/collections/')

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        original_content = content
        replacements_count = 0

        # 1. Replace project_id
        if project_id and project_id in content:
            count = content.count(project_id)
            content = content.replace(project_id, PROJECT_ID_PLACEHOLDER)
            replacements_count += count

        # 2. Replace ge_app_id
        if ge_app_id and ge_app_id in content:
            count = content.count(ge_app_id)
            content = content.replace(ge_app_id, GE_APP_ID_PLACEHOLDER)
            replacements_count += count

        # 3. Anonymize GCP project numbers in resource URIs if any
        if proj_num_pattern.search(content):
            content, num_count = proj_num_pattern.subn(
                f'projects/{PROJECT_NUM_PLACEHOLDER}/locations/global/collections/',
                content
            )
            replacements_count += num_count

        # 4. Enhance comments in config.yaml to guide user
        if fpath.name == "config.yaml":
            # Add helpful comment next to project_id placeholder
            content = re.sub(
                r'project_id:\s*["\']?' + re.escape(PROJECT_ID_PLACEHOLDER) + r'["\']?.*',
                f'project_id: "{PROJECT_ID_PLACEHOLDER}" # Enter your Google Cloud Project ID here (e.g., my-gcp-project-123)',
                content
            )
            # Add helpful comment next to ge_app_id placeholder
            content = re.sub(
                r'ge_app_id:\s*["\']?' + re.escape(GE_APP_ID_PLACEHOLDER) + r'["\']?.*',
                f'ge_app_id: "{GE_APP_ID_PLACEHOLDER}" # Enter your Gemini Enterprise App/Engine ID here (e.g., gemini-enterprise-12345678)',
                content
            )

        if content != original_content:
            modifications[fpath] = replacements_count
            if not dry_run:
                fpath.write_text(content, encoding="utf-8")

    return modifications


def restore_files(
    root_dir: Path,
    new_project_id: str,
    new_ge_app_id: str,
    dry_run: bool = False,
) -> Dict[Path, int]:
    """
    Replaces placeholders with actual user credentials across all files.
    """
    files = get_files_to_process(root_dir)
    modifications = {}

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        original_content = content
        replacements_count = 0

        if new_project_id and PROJECT_ID_PLACEHOLDER in content:
            count = content.count(PROJECT_ID_PLACEHOLDER)
            content = content.replace(PROJECT_ID_PLACEHOLDER, new_project_id)
            replacements_count += count

        if new_ge_app_id and GE_APP_ID_PLACEHOLDER in content:
            count = content.count(GE_APP_ID_PLACEHOLDER)
            content = content.replace(GE_APP_ID_PLACEHOLDER, new_ge_app_id)
            replacements_count += count

        if content != original_content:
            modifications[fpath] = replacements_count
            if not dry_run:
                fpath.write_text(content, encoding="utf-8")

    return modifications


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask or restore sensitive GCP credentials across project files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="GCP Project ID to mask (or new ID to restore). Defaults to extracting from config.yaml.",
    )
    parser.add_argument(
        "--app-id",
        type=str,
        default=os.environ.get("GE_APP_ID"),
        help="Gemini Enterprise App ID to mask (or new ID to restore). Defaults to extracting from config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report occurrences without actually writing files.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore mode: replace placeholders with --project-id and --app-id.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = find_repo_root()
    config_file = root_dir / "config.yaml"

    print("=" * 72)
    print("🔒 AlphaEvolve Project Credentials Sanitization Utility")
    print(f"Repository Root: {root_dir}")
    print("=" * 72)

    if args.restore:
        # Restore mode
        target_pid = args.project_id
        target_app = args.app_id

        # If not provided via CLI, attempt loading from .credentials.backup
        if not target_pid or not target_app:
            backup_pid, backup_app = load_credentials_backup(root_dir)
            target_pid = target_pid or backup_pid
            target_app = target_app or backup_app

        if not target_pid or not target_app:
            print("❌ Error: Both project_id and ge_app_id are required when using --restore.")
            print("   Please provide credentials via CLI arguments:")
            print("     python3 scripts/mask_credentials.py --restore --project-id <PID> --app-id <APP_ID>")
            print("     make unmask PROJECT_ID=<PID> APP_ID=<APP_ID>")
            print(f"   Or ensure local {BACKUP_FILE_NAME} exists from a previous `make mask` execution.")
            sys.exit(1)

        print(f"Mode: RESTORE credentials into placeholders")
        print(f"Target Project ID: {target_pid}")
        print(f"Target App ID:     {target_app}")
        if args.dry_run:
            print("[DRY-RUN MODE] No files will be modified.")

        mods = restore_files(root_dir, target_pid, target_app, dry_run=args.dry_run)
        print(f"\nModified {len(mods)} files:")
        for fpath, count in mods.items():
            rel = fpath.relative_to(root_dir)
            print(f"  - {rel} ({count} replacements)")

        print("\n✅ Restore completed successfully!")
        return

    # Masking mode
    extracted_pid, extracted_app = extract_current_credentials(config_file)
    project_id = args.project_id or extracted_pid
    ge_app_id = args.app_id or extracted_app

    if not project_id and not ge_app_id:
        print("ℹ️ No active credentials detected in config.yaml.")
        print("   The configuration file appears to already be masked with placeholders.")
        print(f"   Placeholder for Project ID: {PROJECT_ID_PLACEHOLDER}")
        print(f"   Placeholder for App ID:     {GE_APP_ID_PLACEHOLDER}")
        print("\nIf you want to mask custom strings, pass --project-id and --app-id explicitly.")
        sys.exit(0)

    # Obfuscate preview for safety output
    def _obfuscate(s: Optional[str]) -> str:
        if not s:
            return "(None)"
        if len(s) <= 6:
            return s[0] + "***" + s[-1]
        return s[:3] + "***" + s[-3:]

    print("Detected credentials to mask:")
    print(f"  • Project ID: {_obfuscate(project_id)}  --> {PROJECT_ID_PLACEHOLDER}")
    print(f"  • GE App ID:  {_obfuscate(ge_app_id)}   --> {GE_APP_ID_PLACEHOLDER}")

    if args.dry_run:
        print("\n[DRY-RUN MODE] Previewing replacements without writing files...")

    mods = mask_files(root_dir, project_id, ge_app_id, dry_run=args.dry_run)

    if not args.dry_run and project_id and ge_app_id:
        backup_file = save_credentials_backup(root_dir, project_id, ge_app_id)
        print(f"\n💾 Saved credentials backup to local {backup_file.name} (protected by .gitignore)")

    print(f"\n{'[DRY-RUN] Would modify' if args.dry_run else 'Successfully modified'} {len(mods)} files:")
    for fpath, count in mods.items():
        rel = fpath.relative_to(root_dir)
        print(f"  ✔ {rel} ({count} occurrences replaced)")

    print("\n" + "=" * 72)
    print("📋 Project Delivery Safety Checklist:")
    print("  1. Credentials masked with `<YOUR_GCP_PROJECT_ID>` and `<YOUR_GE_APP_ID>`.")
    print(f"  2. Credentials safely backed up to {BACKUP_FILE_NAME} (never committed to git).")
    print("  3. In config.yaml, comments now explicitly prompt recipient to fill in their own info.")
    print("  4. Before packing or sending to client, remember to delete the virtual environment:")
    print("     $ rm -rf venv/")
    print("  5. Inform recipient to run `make setup` (or manual venv creation) and `make auth`.")
    print("=" * 72)


if __name__ == "__main__":
    main()
