#!/usr/bin/env python3
"""
Step 1 — clone the source repositories listed in the repo CSV.

Shared by both pipelines: run it once, then step 2, then either (or both) of
the step-3/4/5 chains.

Each repo lands in <work_dir>/repos/{repo_id}_{repo_name}. Clones are shallow
(--depth 1) and already-present directories are skipped, so the script is safe
to re-run after an interrupted download.

The CSV's `sample_id` column is the repo's id. Step 2 assigns a *different*
`sample_id` per source file, so this one is carried as `repo_id` from here on —
see REPO_ID_COLUMN below.

    python common/step_1_download_repos.py
    python common/step_1_download_repos.py --repo-csv my_repos.csv --jobs 8
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import add_config_arg, derive_paths, load_config  # noqa: E402

# The CSV column that identifies a repository.
REPO_ID_COLUMN = "sample_id"


def parse_repo_url(url: str):
    """Extract the repository name from a GitHub URL."""
    parsed = urlparse(url.strip())
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 2:
        return path_parts[-1]
    return None


def clone_repo(repo_id: str, repo_url: str, target_dir: Path, timeout: int = 300) -> str:
    """Clone one repository. Returns 'ok', 'skip' or 'fail'."""
    repo_name = parse_repo_url(repo_url)
    if not repo_name:
        print(f"  [ERROR] Cannot parse repo name from URL: {repo_url}")
        return "fail"

    folder_name = f"{repo_id}_{repo_name}"
    target_path = target_dir / folder_name

    if target_path.exists():
        print(f"  [SKIP] Already exists: {folder_name}")
        return "skip"

    try:
        print(f"  [CLONE] {repo_url} -> {folder_name}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url.strip(), str(target_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            print(f"  [OK] {folder_name}")
            return "ok"
        print(f"  [ERROR] Failed to clone {folder_name}: {result.stderr.strip()}")
        return "fail"

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout while cloning {folder_name}")
        return "fail"
    except Exception as e:
        print(f"  [ERROR] Exception while cloning {folder_name}: {e}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(description="Clone the repositories listed in the repo CSV.")
    add_config_arg(parser)
    parser.add_argument("--repo-csv", default=None, help="Override paths.repo_csv")
    parser.add_argument("--repos-dir", default=None, help="Override <work_dir>/repos")
    parser.add_argument("--jobs", "-j", type=int, default=4, help="Parallel clones (default: 4)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-clone timeout in seconds")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = derive_paths(cfg)

    csv_file = Path(args.repo_csv) if args.repo_csv else paths["repo_csv"]
    target_dir = Path(args.repos_dir) if args.repos_dir else paths["repos_dir"]

    if not csv_file.exists():
        sys.exit(f"Error: repo CSV not found: {csv_file}")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo CSV:         {csv_file}")
    print(f"Target directory: {target_dir}")
    print()

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and REPO_ID_COLUMN not in reader.fieldnames:
            sys.exit(
                f"Error: {csv_file} has no '{REPO_ID_COLUMN}' column "
                f"(found: {', '.join(reader.fieldnames)}).\n"
                "See the README's 'Bringing your own repositories' section."
            )
        for row in reader:
            repo_id = row.get(REPO_ID_COLUMN, "").strip()
            repo_url = row.get("repository_url", "").strip()
            if not repo_id or not repo_url:
                print(f"  [SKIP] Invalid row: {row}")
                continue
            rows.append((repo_id, repo_url))

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(clone_repo, repo_id, repo_url, target_dir, args.timeout)
            for repo_id, repo_url in rows
        ]
        for fut in as_completed(futures):
            counts[fut.result()] += 1

    print("\n" + "=" * 50)
    print("Download Summary:")
    print(f"  Cloned:  {counts['ok']}")
    print(f"  Skipped: {counts['skip']} (already present)")
    print(f"  Failed:  {counts['fail']}")
    print(f"  Total:   {sum(counts.values())}")
    print("=" * 50)


if __name__ == "__main__":
    main()
