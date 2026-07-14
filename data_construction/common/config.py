#!/usr/bin/env python3
"""
Central configuration loader.

Every step script calls `load_config()` to get a merged view of config.yaml +
built-in defaults, and `derive_paths()` to turn `paths.work_dir` into the
concrete artifact paths the pipeline hands from one step to the next.

The only two things a user must set are `paths.repo_csv` and `paths.work_dir`;
everything downstream is derived, so no absolute path is ever baked into a
script.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to read config.yaml. Install it with:\n"
        "    pip install pyyaml"
    ) from exc


# Repository root = the directory containing config.yaml (one level above this file).
RELEASE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = RELEASE_ROOT / "config.yaml"


DEFAULTS: Dict[str, Any] = {
    "paths": {
        "repo_csv": "data/example_repos.csv",
        "work_dir": "workdir",
    },
    "llm": {
        "model": "gemini-3-flash-preview",
        "fim_temperature": 0.7,
        "critique_temperature": 0.3,
        "wait_seconds": 0.5,
        "max_retries": 3,
        "retry_delay": 5.0,
        "price_per_1m_input_tokens": 0.50,
        "price_per_1m_output_tokens": 3.00,
    },
    "sharding": {
        "total_shards": 200,
        "shard_start": 1,
        "shard_end": 200,
        "concurrency": 1,
    },
    "selection": {},
    "filters": {
        "single": {"min_individual_score": 3, "min_overall_score": 4},
        "multi": {
            "min_per_func_score": 3,
            "min_group_overall_score": 4,
            "min_coherence_score": 3,
        },
    },
    "wandb": {
        "enabled": False,
        "project": "fim-dataset-curation",
        "entity": "",
        "run_name": "",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load config.yaml and merge it over the built-in defaults.

    `path` may be None (use the config.yaml next to this package), a path to a
    different YAML file, or overridden by the FIM_CONFIG environment variable.
    """
    cfg_path = Path(path or os.getenv("FIM_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, user_cfg)
    cfg["_config_path"] = str(cfg_path)
    cfg["_config_dir"] = str(cfg_path.parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str) -> Path:
    """Resolve a config path: absolute stays as-is, relative hangs off config.yaml."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (Path(cfg.get("_config_dir", RELEASE_ROOT)) / p).resolve()


def derive_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Turn `paths.work_dir` into every artifact path the pipeline uses.

    Naming matters: step 5 finds step 4's output by globbing for the checkpoint
    stem, so the two must stay in sync. Change a name here and both ends move
    together.
    """
    work = resolve_path(cfg, cfg["paths"]["work_dir"])

    single = work / "single_function"
    multi = work / "multi_function"

    return {
        "work_dir": work,
        "repo_csv": resolve_path(cfg, cfg["paths"]["repo_csv"]),
        # ---- shared stages ----
        "repos_dir": work / "repos",
        "extracted_files": work / "extracted_python_files.json",
        # ---- single-function pipeline ----
        "single_dir": single,
        # step 3 writes <stem>.json and <stem>_functions.json
        "single_step3_out": single / "step_3_selected.json",
        "single_step3_functions": single / "step_3_selected_functions.json",
        # step 4 writes <stem>_shard{i}_of_{N}.json and
        #              <stem>_checkpoint_shard{i}_of_{N}.json
        "single_step4_out": single / "step_4_fim_critique.json",
        "single_step4_checkpoint_glob": "step_4_fim_critique_checkpoint*.json",
        # step 5
        "single_sft_out": single / "sft" / "single_function_fim_sft.jsonl",
        "single_sft_stats": single / "sft" / "stats.json",
        # ---- multi-function pipeline ----
        "multi_dir": multi,
        # step 3 writes <stem>.json and <stem>_groups.json
        "multi_step3_out": multi / "step_3_selected.json",
        "multi_step3_groups": multi / "step_3_selected_groups.json",
        "multi_step4_out": multi / "step_4_multi_fim.json",
        "multi_step4_checkpoint_glob": "step_4_multi_fim_checkpoint*.json",
        "multi_sft_pairs": multi / "sft" / "multi_function_fim_sft_pairs.jsonl",
        "multi_sft_triples": multi / "sft" / "multi_function_fim_sft_triples.jsonl",
        "multi_sft_stats": multi / "sft" / "stats.json",
    }


def add_config_arg(parser) -> None:
    """Add the standard --config flag to an argparse parser."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: the config.yaml next to this release, "
             "or $FIM_CONFIG)",
    )


def selection_config(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The `selection` block, or None when empty (dep_graph then uses its own defaults)."""
    sel = cfg.get("selection") or {}
    return dict(sel) if sel else None
