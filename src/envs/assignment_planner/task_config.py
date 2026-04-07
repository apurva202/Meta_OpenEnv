"""
task_config.py
--------------
Canonical task-episode configurations for the Assignment & Bug-Fix Planner Agent.

Each difficulty section now contains **5 variants**.  At runtime the inference
script randomly samples N tasks from each section (default N=1 → 3 tasks total,
matching the original behaviour; pass --random 5 to get 5 per section).

Naming convention:
  easy_1   … easy_5
  medium_1 … medium_5
  hard_1   … hard_5
"""

import random as _random
from typing import Any, Dict, List

TaskDef = Dict[str, Any]

# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ════════════════════════════════════════════════════════════════════════
    # EASY  –  2 tasks, generous timeline.  Total work ≤ daily capacity.
    # ════════════════════════════════════════════════════════════════════════

    # easy_1 – HIGH bug + feature.  Total=6h, cap=18h.
    "easy_1": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Fix high-severity login bug",
             "type": "bug", "severity": "high", "deadline": 3, "estimated_hours": 2.0},
            {"id": 1, "name": "Implement basic dashboard feature",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # easy_2 – HIGH bug + small feature.  Total=5h, cap=12h.
    "easy_2": {
        "max_days": 2, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Fix null-pointer crash on startup",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 3.0},
            {"id": 1, "name": "Add welcome onboarding wizard",
             "type": "feature", "severity": None, "deadline": 2, "estimated_hours": 2.0},
        ],
    },

    # easy_3 – MEDIUM bug + feature.  Total=6h, cap=18h.
    "easy_3": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Fix broken search index",
             "type": "bug", "severity": "medium", "deadline": 3, "estimated_hours": 2.0},
            {"id": 1, "name": "Add paginated results view",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # easy_4 – LOW bug + feature, very generous cap.
    "easy_4": {
        "max_days": 4, "daily_capacity": 8.0,
        "tasks": [
            {"id": 0, "name": "Fix display glitch in settings panel",
             "type": "bug", "severity": "low", "deadline": 4, "estimated_hours": 1.0},
            {"id": 1, "name": "Build CSV data-export button",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 4.0},
        ],
    },

    # easy_5 – review + feature only (no bugs; score_bugs defaults 1.0).
    "easy_5": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Review PR: auth module refactor",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 2.0},
            {"id": 1, "name": "Build user avatar upload",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # ════════════════════════════════════════════════════════════════════════
    # MEDIUM  –  3 tasks, one tight deadline.  Capacity is reachable but
    #            requires correct prioritisation.
    # ════════════════════════════════════════════════════════════════════════

    # medium_1 – original: HIGH bug (deadline 2) + feature + refactor.
    "medium_1": {
        "max_days": 4, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Fix high-severity API crash on /checkout",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 3.0},
            {"id": 1, "name": "Implement profile settings page",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 3.0},
            {"id": 2, "name": "Refactor auth module (low-priority cleanup)",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 5.0},
        ],
    },

    # medium_2 – HIGH security bug (deadline 1!) + 2 features.
    "medium_2": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: auth bypass vulnerability (CVE)",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 5.0},
            {"id": 1, "name": "Build password-reset email flow",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 3.0},
            {"id": 2, "name": "Review PR: OAuth2 integration",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 3.0},
        ],
    },

    # medium_3 – 2 medium bugs + 1 feature.  Agent must order by severity.
    "medium_3": {
        "max_days": 4, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "Fix broken CSV export (medium sev)",
             "type": "bug", "severity": "medium", "deadline": 2, "estimated_hours": 3.0},
            {"id": 1, "name": "Fix incorrect pagination offset",
             "type": "bug", "severity": "medium", "deadline": 3, "estimated_hours": 2.0},
            {"id": 2, "name": "Build analytics dashboard widget",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 5.0},
        ],
    },

    # medium_4 – HIGH bug + 2 features; total=11h, cap=18h but tight daily.
    "medium_4": {
        "max_days": 3, "daily_capacity": 5.0,
        "tasks": [
            {"id": 0, "name": "Fix data-loss bug on import job",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 4.0},
            {"id": 1, "name": "Feature: real-time notifications",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 3.0},
            {"id": 2, "name": "Feature: dark mode toggle",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # medium_5 – HIGH bug (deadline 1) + feature + review with tight cap.
    "medium_5": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: session hijack vector",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 4.0},
            {"id": 1, "name": "Feature: audit log viewer",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 5.0},
            {"id": 2, "name": "Review: security hardening PR",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 3.0},
        ],
    },

    # ════════════════════════════════════════════════════════════════════════
    # HARD  –  5 tasks, total work > capacity.  Must triage ruthlessly.
    # ════════════════════════════════════════════════════════════════════════

    # hard_1 – original: 2 critical bugs + medium bug + large feature + review.
    "hard_1": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: memory leak in data pipeline",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 4.0},
            {"id": 1, "name": "CRITICAL: race condition in websocket handler",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 4.0},
            {"id": 2, "name": "Fix medium: incorrect pagination offset",
             "type": "bug", "severity": "medium", "deadline": 3, "estimated_hours": 2.0},
            {"id": 3, "name": "Migrate legacy endpoints to REST v2",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 7.0},
            {"id": 4, "name": "Review security audit report",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # hard_2 – 3 critical bugs back-to-back deadlines + 2 features.
    "hard_2": {
        "max_days": 4, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: payments service crash",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 5.0},
            {"id": 1, "name": "CRITICAL: data corruption on checkout",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 4.0},
            {"id": 2, "name": "CRITICAL: deadlock in order processing",
             "type": "bug", "severity": "high", "deadline": 3, "estimated_hours": 3.0},
            {"id": 3, "name": "Build inventory reconciliation feature",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 5.0},
            {"id": 4, "name": "Review supplier API integration",
             "type": "review", "severity": None, "deadline": 4, "estimated_hours": 5.0},
        ],
    },

    # hard_3 – Security sprint: 2 critical security bugs + medium + feature + review.
    "hard_3": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: SQL injection in search endpoint",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 3.0},
            {"id": 1, "name": "CRITICAL: XSS in comments system",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 3.0},
            {"id": 2, "name": "Medium: CORS misconfiguration",
             "type": "bug", "severity": "medium", "deadline": 3, "estimated_hours": 2.0},
            {"id": 3, "name": "Build API rate-limiting module",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 6.0},
            {"id": 4, "name": "Review penetration test findings",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },

    # hard_4 – Infrastructure: DB + cache bugs + massive migration + review.
    "hard_4": {
        "max_days": 4, "daily_capacity": 7.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: DB connection pool exhaustion",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 4.0},
            {"id": 1, "name": "CRITICAL: cache invalidation storm",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 5.0},
            {"id": 2, "name": "Medium: slow queries on report endpoint",
             "type": "bug", "severity": "medium", "deadline": 4, "estimated_hours": 3.0},
            {"id": 3, "name": "Migrate monolith to microservices (phase 1)",
             "type": "feature", "severity": None, "deadline": 4, "estimated_hours": 8.0},
            {"id": 4, "name": "Review Kubernetes deployment configs",
             "type": "review", "severity": None, "deadline": 4, "estimated_hours": 4.0},
        ],
    },

    # hard_5 – ML sprint: model bugs + data pipeline + training feature.
    "hard_5": {
        "max_days": 3, "daily_capacity": 6.0,
        "tasks": [
            {"id": 0, "name": "CRITICAL: model inference crash on GPU 0",
             "type": "bug", "severity": "high", "deadline": 1, "estimated_hours": 4.0},
            {"id": 1, "name": "CRITICAL: training job OOM on large batches",
             "type": "bug", "severity": "high", "deadline": 2, "estimated_hours": 5.0},
            {"id": 2, "name": "Fix medium: NaN loss in fine-tuning loop",
             "type": "bug", "severity": "medium", "deadline": 3, "estimated_hours": 2.0},
            {"id": 3, "name": "Implement distributed training launcher",
             "type": "feature", "severity": None, "deadline": 3, "estimated_hours": 6.0},
            {"id": 4, "name": "Review data-pipeline architecture doc",
             "type": "review", "severity": None, "deadline": 3, "estimated_hours": 4.0},
        ],
    },
}

# ---------------------------------------------------------------------------
# Difficulty sections  (used for random sampling)
# ---------------------------------------------------------------------------
DIFFICULTY_SECTIONS: Dict[str, List[str]] = {
    "easy":   ["easy_1",   "easy_2",   "easy_3",   "easy_4",   "easy_5"],
    "medium": ["medium_1", "medium_2", "medium_3", "medium_4", "medium_5"],
    "hard":   ["hard_1",   "hard_2",   "hard_3",   "hard_4",   "hard_5"],
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def list_task_ids() -> List[str]:
    """Return all registered task_id strings."""
    return list(TASK_CONFIGS.keys())


def list_task_ids_by_difficulty(difficulty: str) -> List[str]:
    """
    Return all task IDs for a given difficulty level.

    Parameters
    ----------
    difficulty : str
        One of ``'easy'``, ``'medium'``, or ``'hard'``.

    Raises
    ------
    KeyError
        If the difficulty is not recognised.
    """
    if difficulty not in DIFFICULTY_SECTIONS:
        raise KeyError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(DIFFICULTY_SECTIONS)}"
        )
    return list(DIFFICULTY_SECTIONS[difficulty])


def sample_tasks(n: int = 1, seed: int = None) -> List[str]:
    """
    Randomly sample ``n`` task IDs from **each** difficulty section.

    Parameters
    ----------
    n    : int
        Number of tasks to pick from each section (1 ≤ n ≤ 5).
        Default is 1, which replicates the original 3-task behaviour.
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    List[str]
        Sampled task IDs ordered as: easy ... | medium ... | hard ...
    """
    n = max(1, min(n, 5))
    rng = _random.Random(seed)
    result: List[str] = []
    for diff in ("easy", "medium", "hard"):
        pool = DIFFICULTY_SECTIONS[diff]
        result.extend(rng.sample(pool, n))
    return result


def get_config(task_id: str) -> Dict[str, Any]:
    """
    Retrieve the full configuration for a given task_id.

    Raises
    ------
    KeyError
        If ``task_id`` is not found in TASK_CONFIGS.
    """
    if task_id not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list_task_ids()}"
        )
    return TASK_CONFIGS[task_id]
