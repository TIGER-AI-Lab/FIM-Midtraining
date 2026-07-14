#!/usr/bin/env python3
"""
Step 2 — flatten the cloned repos into one JSON file, one record per .py file.

Shared by both pipelines. The output of this step is the input of both
single_function/step_3 and multi_function/step_3, so it only needs to run once.

Each record carries the file's source plus the repo metadata from the CSV:

    {
      "sample_id": 0,                     # per-FILE id, assigned here
      "repo_id": "0",                     # the CSV's sample_id column
      "repository_url": "https://github.com/...",
      "file_path": "/path/inside/repo.py",
      "line_num": 214,                    # lines in the file
      "func_num": 12,                     # functions + methods (-1 = unparseable)
      "category": ..., "description": ..., "notes": ..., "license": ...,
      "code_content": "..."               # full source
    }

Note the two ids. The CSV's `sample_id` identifies a *repository*; this step
re-keys it as `repo_id` and mints a fresh `sample_id` per source file, because
everything downstream identifies a FIM record by (sample_id, func_name,
start_line). If a repo's id leaked into the per-file field, every file in that
repo would share an id and distinct functions would collide in step 4's
checkpoint dedup.

    python common/step_2_extract_python_files.py
"""

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import add_config_arg, derive_paths, load_config  # noqa: E402

# Directories that never contain source worth mining.
SKIP_DIRS = {
    '__pycache__', 'node_modules', 'venv', 'env', '.git',
    'build', 'dist', 'egg-info', '.eggs', '.tox',
}

# The CSV column that identifies a repository (see the module docstring on why
# it becomes `repo_id` rather than staying `sample_id`).
REPO_ID_COLUMN = 'sample_id'

# The metadata columns copied verbatim from the CSV onto every record. A column
# that isn't in your CSV simply comes through empty.
METADATA_COLUMNS = (
    'repository_url', 'category', 'description', 'notes', 'license',
)


def load_repo_metadata(csv_file: Path) -> dict:
    """Load the repo CSV into a dict keyed by repo_id."""
    metadata = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and REPO_ID_COLUMN not in reader.fieldnames:
            sys.exit(
                f"Error: {csv_file} has no '{REPO_ID_COLUMN}' column "
                f"(found: {', '.join(reader.fieldnames)}).\n"
                "See the README's 'Bringing your own repositories' section."
            )
        for row in reader:
            repo_id = row.get(REPO_ID_COLUMN, '').strip()
            if not repo_id:
                continue
            entry = {'repo_id': repo_id}
            for col in METADATA_COLUMNS:
                entry[col] = row.get(col, '').strip()
            metadata[repo_id] = entry
    return metadata


def find_repo_folders(repos_dir: Path) -> dict:
    """Map repo_id -> folder path, using the {repo_id}_{repo_name} convention."""
    repo_folders = {}
    for folder in repos_dir.iterdir():
        if folder.is_dir() and '_' in folder.name:
            repo_folders[folder.name.split('_')[0]] = folder
    return repo_folders


def read_python_file(file_path: Path) -> Optional[str]:
    """Read a Python file, trying a few encodings before giving up."""
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [WARN] Error reading {file_path}: {e}")
            return None
    print(f"  [WARN] Could not decode {file_path} with any encoding")
    return None


def count_functions(content: str) -> int:
    """Count functions and methods. Returns -1 when the file does not parse."""
    if not content:
        return 0
    try:
        tree = ast.parse(content)
    except Exception:
        return -1
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def extract_python_files(repo_folder: Path, repo_metadata: dict, id_counter: int):
    """Extract every .py file under one repo. Returns (records, next_id)."""
    records = []
    current_id = id_counter

    for root, dirs, files in os.walk(repo_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]

        for file in files:
            if not file.endswith('.py'):
                continue

            file_path = Path(root) / file
            relative_path = '/' + str(file_path.relative_to(repo_folder)).replace('\\', '/')

            content = read_python_file(file_path)
            if content is None:
                continue

            record = {
                'sample_id': current_id,
                'repo_id': repo_metadata['repo_id'],
                'file_path': relative_path,
                'line_num': len(content.splitlines()),
                'func_num': count_functions(content),
                'code_content': content,
            }
            for col in METADATA_COLUMNS:
                record[col] = repo_metadata.get(col, '')

            records.append(record)
            current_id += 1

    return records, current_id


def main():
    parser = argparse.ArgumentParser(
        description="Extract Python files from the cloned repos into a single JSON file."
    )
    add_config_arg(parser)
    parser.add_argument("--repo-csv", default=None, help="Override paths.repo_csv")
    parser.add_argument("--repos-dir", default=None, help="Override <work_dir>/repos")
    parser.add_argument("--output", "-o", default=None,
                        help="Override <work_dir>/extracted_python_files.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)

    csv_file = Path(args.repo_csv) if args.repo_csv else paths["repo_csv"]
    repos_dir = Path(args.repos_dir) if args.repos_dir else paths["repos_dir"]
    output_path = Path(args.output) if args.output else paths["extracted_files"]

    if not csv_file.exists():
        sys.exit(f"Error: repo CSV not found: {csv_file}")
    if not repos_dir.exists():
        sys.exit(f"Error: repos directory not found: {repos_dir}\nRun step 1 first.")

    print(f"Repo CSV:    {csv_file}")
    print(f"Repos dir:   {repos_dir}")
    print(f"Output:      {output_path}\n")

    repo_metadata = load_repo_metadata(csv_file)
    print(f"Loading repository metadata... {len(repo_metadata)} repos in CSV")

    repo_folders = find_repo_folders(repos_dir)
    print(f"Scanning repository folders... {len(repo_folders)} folders found\n")

    all_records = []
    repos_processed = 0
    repos_not_found = 0
    id_counter = 0
    parse_error_count = 0

    for repo_id, metadata in repo_metadata.items():
        if repo_id not in repo_folders:
            print(f"[{repo_id}] Folder not found, skipping")
            repos_not_found += 1
            continue

        repo_folder = repo_folders[repo_id]
        print(f"[{repo_id}] Processing: {repo_folder.name}")

        records, id_counter = extract_python_files(repo_folder, metadata, id_counter)
        all_records.extend(records)

        batch_errors = sum(1 for r in records if r['func_num'] == -1)
        parse_error_count += batch_errors
        print(f"  Extracted {len(records)} Python files"
              + (f" ({batch_errors} parse errors)" if batch_errors else ""))
        repos_processed += 1

    print(f"\nWriting {len(all_records)} records to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("Extraction Summary:")
    print(f"  Repos in CSV:    {len(repo_metadata)}")
    print(f"  Repos processed: {repos_processed}")
    print(f"  Repos not found: {repos_not_found}")
    print(f"  Python files:    {len(all_records)}")
    print(f"  Parse errors:    {parse_error_count} (func_num = -1)")
    print(f"  Output file:     {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
