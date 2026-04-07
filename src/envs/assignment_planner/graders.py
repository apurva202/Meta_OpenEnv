"""
graders.py
----------
Three deterministic graders (easy / medium / hard) that evaluate a complete
episode trajectory and return a 0.0–1.0 score.

A **trajectory** is a list of ``State`` snapshots collected at every step
during an episode (including the terminal state).  The graders are *pure
functions*: given the same trajectory they always return the same score, with
no side-effects or randomness.

Scoring formula (per grader)
-----------------------------
    final_score = clip(
        alpha  * score_bugs
      + beta   * score_features
      - gamma  * workload_ratio
      - delta  * bug_ignored
      + bonus,
        0.0, 1.0
    )

Where:
  score_bugs     – fraction of high-severity bugs finished before their deadline.
  score_features – fraction of feature tasks finished (any time before episode end).
  workload_ratio – fraction of *completed* days where hours_spent > 8h.
  bug_ignored    – 1 if any high-severity bug was completely untouched until the
                   final day of the episode, 0 otherwise.
  bonus          – optional bonus for "burnout-free" or "balanced" behaviour.

Each grader sets its own (α, β, γ, δ, bonus) coefficients.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from .models import State, Task

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Trajectory = List[State]
"""List of State snapshots, one per step (including the terminal step)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hours_spent_on_day(trajectory: Trajectory, target_day: int) -> float:
    """
    Estimate total hours spent on ``target_day`` by looking at reductions in
    ``hours_left_today`` across consecutive states that share the same day.

    We only count within-day spend (not the rollover replenishment).
    """
    day_states = [s for s in trajectory if s.day == target_day]
    if not day_states:
        return 0.0

    # hours_left decrements within a day; biggest drop = most hours worked
    max_hours = max(s.hours_left_today for s in day_states)
    min_hours = min(s.hours_left_today for s in day_states)
    return round(max_hours - min_hours, 4)


def _task_finished_on_time(task_id: int, trajectory: Trajectory) -> bool:
    """
    Return True if the task with ``task_id`` has status "done" in some state
    *before* its deadline expired (i.e. when the state's deadline for that
    task was still > 0 and the task was just marked done).

    Strategy: scan through the trajectory; the moment a task's status
    transitions to "done", check if its deadline is still >= 0 in that state.
    (Deadline is decremented at end-of-day, so finishing just before midnight
    counts as on-time.)
    """
    prev_status = "not_started"
    for state in trajectory:
        task = next((t for t in state.tasks if t.id == task_id), None)
        if task is None:
            continue
        if task.status == "done" and prev_status != "done":
            # Transition to "done" detected – check deadline at this moment
            return task.deadline >= 0
        prev_status = task.status
    return False  # never finished


def _bug_ignored_until_last_day(
    task_id: int, trajectory: Trajectory, last_day: int
) -> bool:
    """
    Return True if task ``task_id`` (a high-severity bug) was completely
    untouched (status == "not_started") until ``last_day``.

    "Untouched until last day" means every state whose ``day < last_day``
    shows the task as "not_started".
    """
    for state in trajectory:
        if state.day < last_day:
            task = next((t for t in state.tasks if t.id == task_id), None)
            if task and task.status != "not_started":
                return False  # was touched before the last day
    return True


def _trajectory_to_stats(
    trajectory: Trajectory,
) -> Dict[str, Any]:
    """
    Compute a rich stat dictionary from a recorded episode trajectory.

    Returns
    -------
    dict with keys:
        ``score_bugs``          – float [0, 1]
        ``score_features``      – float [0, 1]
        ``workload_ratio``      – float [0, 1] (days with >8h / total days)
        ``bug_ignored``         – int  0 or 1
        ``n_high_bugs``         – int
        ``n_features``          – int
        ``days_elapsed``        – int
        ``daily_hours_spent``   – dict[int, float]
        ``burnout_free``        – bool (no day exceeded 8h)
        ``balanced_work``       – bool (both bugs and features touched each day)
    """
    if not trajectory:
        return {
            "score_bugs": 0.0,
            "score_features": 0.0,
            "workload_ratio": 0.0,
            "bug_ignored": 0,
            "n_high_bugs": 0,
            "n_features": 0,
            "days_elapsed": 0,
            "daily_hours_spent": {},
            "burnout_free": True,
            "balanced_work": False,
        }

    terminal = trajectory[-1]
    last_day = terminal.day

    # ── Task metadata from the initial state ──────────────────────────────
    initial_tasks = trajectory[0].tasks
    high_bugs = [t for t in initial_tasks if t.type == "bug" and t.severity == "high"]
    features = [t for t in initial_tasks if t.type == "feature"]

    # ── score_bugs: fraction of high-severity bugs finished on time ───────
    bugs_on_time = sum(
        1 for t in high_bugs if _task_finished_on_time(t.id, trajectory)
    )
    score_bugs = bugs_on_time / len(high_bugs) if high_bugs else 1.0

    # ── score_features: fraction of features finished (any time) ─────────
    features_done = sum(
        1
        for t in features
        if any(
            s_t.status == "done"
            for s in trajectory
            for s_t in s.tasks
            if s_t.id == t.id
        )
    )
    score_features = features_done / len(features) if features else 1.0

    # ── workload_ratio ─────────────────────────────────────────────────────
    # Compute hours spent per day; mark days with spend > 8h as "overloaded".
    # We determine days from the states present in the trajectory.
    days_seen = sorted({s.day for s in trajectory})
    daily_hours: Dict[int, float] = {}
    for d in days_seen:
        daily_hours[d] = _hours_spent_on_day(trajectory, d)

    overloaded_days = sum(1 for h in daily_hours.values() if h > 8.0)
    total_days = len(days_seen) if days_seen else 1
    workload_ratio = overloaded_days / total_days

    # ── bug_ignored ───────────────────────────────────────────────────────
    bug_ignored = int(
        any(
            _bug_ignored_until_last_day(t.id, trajectory, last_day)
            for t in high_bugs
        )
    )

    # ── burnout_free ──────────────────────────────────────────────────────
    burnout_free = all(h <= 8.0 for h in daily_hours.values())

    # ── balanced_work ─────────────────────────────────────────────────────
    # True if the agent touched at least one bug AND one feature across the
    # episode (not necessarily every day).
    ever_on_bug = False
    ever_on_feature = False
    for state in trajectory:
        for t in state.tasks:
            if t.status in ("in_progress", "done"):
                if t.type == "bug":
                    ever_on_bug = True
                if t.type == "feature":
                    ever_on_feature = True
    balanced_work = ever_on_bug and ever_on_feature

    return {
        "score_bugs": score_bugs,
        "score_features": score_features,
        "workload_ratio": workload_ratio,
        "bug_ignored": bug_ignored,
        "n_high_bugs": len(high_bugs),
        "n_features": len(features),
        "days_elapsed": last_day,
        "daily_hours_spent": daily_hours,
        "burnout_free": burnout_free,
        "balanced_work": balanced_work,
    }


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public grader functions
# ---------------------------------------------------------------------------

def grade_easy(trajectory: Trajectory) -> float:
    """
    Evaluate an *easy* episode trajectory and return a 0.0–1.0 score.

    Weights (α=0.7, β=0.3, γ=0.05, δ=0.1, bonus=0.0)
    ---------------------------------------------------
    - Heavy emphasis on finishing the lone high-severity bug on time.
    - Moderate credit for completing the feature.
    - Very light penalty for overloaded days (easy scenario is lenient).
    - Small penalty if the critical bug is completely ignored until the end.

    Parameters
    ----------
    trajectory : List[State]
        Ordered list of State snapshots from reset() through the final step().

    Returns
    -------
    float
        Scalar score in [0.0, 1.0].
    """
    stats = _trajectory_to_stats(trajectory)

    alpha = 0.70   # weight for high-severity bug completion
    beta  = 0.30   # weight for feature completion
    gamma = 0.05   # penalty coefficient for overloaded days
    delta = 0.10   # penalty if a critical bug is ignored until the last day

    raw = (
        alpha * stats["score_bugs"]
        + beta  * stats["score_features"]
        - gamma * stats["workload_ratio"]
        - delta * stats["bug_ignored"]
    )
    return round(_clip(raw), 4)


def grade_medium(trajectory: Trajectory) -> float:
    """
    Evaluate a *medium* episode trajectory and return a 0.0–1.0 score.

    Weights (α=0.55, β=0.25, γ=0.20, δ=0.20, bonus≤0.10)
    -------------------------------------------------------
    - Still prioritises bugs, but features carry more weight than easy.
    - Moderate burnout penalty (γ raised to 0.20).
    - Stronger ignore penalty (δ=0.20): leaving a critical bug for the last
      day is a red flag at medium difficulty.
    - Up to +0.10 bonus for "balanced work" (touched both bugs and features).

    Parameters
    ----------
    trajectory : List[State]
        Ordered list of State snapshots from reset() through the final step().

    Returns
    -------
    float
        Scalar score in [0.0, 1.0].
    """
    stats = _trajectory_to_stats(trajectory)

    alpha = 0.55
    beta  = 0.25
    gamma = 0.20
    delta = 0.20
    bonus = 0.10 if stats["balanced_work"] else 0.0

    raw = (
        alpha * stats["score_bugs"]
        + beta  * stats["score_features"]
        - gamma * stats["workload_ratio"]
        - delta * stats["bug_ignored"]
        + bonus
    )
    return round(_clip(raw), 4)


def grade_hard(trajectory: Trajectory) -> float:
    """
    Evaluate a *hard* episode trajectory and return a 0.0–1.0 score.

    Weights (α=0.60, β=0.15, γ=0.30, δ=0.25, bonus≤0.15)
    -------------------------------------------------------
    - Critical bugs dominate scoring (α=0.60); features carry less weight
      because the total work exceeds the available capacity (triage required).
    - Strong burnout penalty (γ=0.30): hard scenarios make overloading tempting
      but reward sustainable pacing.
    - Strongest ignore penalty (δ=0.25): completely ignoring a critical bug
      in a high-pressure scenario is a serious failure mode.
    - Up to +0.15 bonus for fully burnout-free execution (no day > 8h worked).

    Parameters
    ----------
    trajectory : List[State]
        Ordered list of State snapshots from reset() through the final step().

    Returns
    -------
    float
        Scalar score in [0.0, 1.0].
    """
    stats = _trajectory_to_stats(trajectory)

    alpha = 0.60
    beta  = 0.15
    gamma = 0.30
    delta = 0.25
    bonus = 0.15 if stats["burnout_free"] else 0.0

    raw = (
        alpha * stats["score_bugs"]
        + beta  * stats["score_features"]
        - gamma * stats["workload_ratio"]
        - delta * stats["bug_ignored"]
        + bonus
    )
    return round(_clip(raw), 4)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

# Built dynamically: any task_id whose prefix matches easy/medium/hard is
# automatically registered with the correct grader.  This means adding new
# variants to task_config.py (easy_2, hard_5, …) requires NO changes here.
from .task_config import list_task_ids as _list_task_ids

def _build_grader_map():
    _prefix_map = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    result = {}
    for tid in _list_task_ids():
        for prefix, fn in _prefix_map.items():
            if tid.startswith(prefix):
                result[tid] = fn
                break
    return result

GRADER_MAP = _build_grader_map()


def grade(task_id: str, trajectory: Trajectory) -> float:
    """
    Dispatch to the appropriate grader based on ``task_id``.

    Parameters
    ----------
    task_id : str
        Must be one of "easy_1", "medium_1", "hard_1".
    trajectory : List[State]
        Episode trajectory.

    Returns
    -------
    float
        Scalar score in [0.0, 1.0].

    Raises
    ------
    KeyError
        If ``task_id`` is not in GRADER_MAP.
    """
    if task_id not in GRADER_MAP:
        raise KeyError(
            f"No grader for task_id '{task_id}'. "
            f"Available: {list(GRADER_MAP.keys())}"
        )
    return GRADER_MAP[task_id](trajectory)
