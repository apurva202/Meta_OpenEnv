#!/usr/bin/env python3
"""
inference.py
============
Baseline inference script for the Assignment & Bug-Fix Planner Agent.

Drives an AI agent (via any OpenAI-compatible LLM endpoint) through all
three task episodes (easy_1, medium_1, hard_1) and prints structured logs
in the required [START] / [STEP] / [END] format.

Modes
-----
HTTP mode  (default):
    Talks to the FastAPI server running on HF Spaces (or locally via Docker).
    Calls POST /reset and POST /step over HTTP.

Local mode (--local flag or USE_LOCAL_ENV=1):
    Bypasses the HTTP server entirely and uses AssignmentPlannerEnv directly.
    Useful for debugging without a running server.

Environment variables
---------------------
API_BASE_URL   : Base URL of the OpenEnv HTTP server
                 (e.g., "https://your-space.hf.space").
                 Required for HTTP mode.
MODEL_NAME     : LLM model identifier
                 (e.g., "meta-llama/Llama-3.1-8B-Instruct").
HF_TOKEN       : API key / Hugging Face token for the LLM endpoint.
USE_LOCAL_ENV  : Set to "1" to skip HTTP and use local Python env.

Usage
-----
# Remote mode (against a running HF Space):
    export API_BASE_URL="https://your-space.hf.space"
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py

# Local mode (no HTTP server needed, uses LLM for decisions):
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    export HF_TOKEN="hf_..."
    USE_LOCAL_ENV=1 python inference.py

# Fully local (heuristic agent, no LLM, no server — for quick sanity checks):
    USE_LOCAL_ENV=1 python inference.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap – so `src.envs.assignment_planner` is importable from
# wherever the script is run (repo root expected).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.envs.assignment_planner.environment import AssignmentPlannerEnv
from src.envs.assignment_planner.graders import grade
from src.envs.assignment_planner.models import Action, Observation, State
from src.envs.assignment_planner.task_config import TASK_CONFIGS, list_task_ids, sample_tasks

# ---------------------------------------------------------------------------
# Logging (structured, prints to stdout for clean HF log capture)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://apurva-22-meta-openenv.hf.space").rstrip("/")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")
USE_LOCAL_ENV: bool = os.getenv("USE_LOCAL_ENV", "0") == "1"

# LLM call settings
MAX_RETRIES: int = 3          # per action
RETRY_DELAY: float = 1.0      # seconds between retries
LLM_TIMEOUT: int = 30         # seconds per LLM call
MAX_TOKENS: int = 256         # keep responses tight
TEMPERATURE: float = 0.2      # low randomness for reproducible actions


# ===========================================================================
# HTTP client helpers (no external deps besides stdlib)
# ===========================================================================

def _http(
    method: str,
    url: str,
    body: Optional[Dict] = None,
    timeout: int = 30,
) -> Dict:
    """
    Minimal HTTP request using stdlib only.

    Parameters
    ----------
    method : "GET" or "POST"
    url    : full URL string
    body   : dict to JSON-serialise as request body (for POST)
    timeout: seconds

    Returns
    -------
    Parsed JSON response as a dict.

    Raises
    ------
    RuntimeError on non-2xx responses.
    """
    data = json.dumps(body).encode() if body is not None else None
    headers: Dict[str, str] = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("detail", str(exc))
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def http_reset(task_id: str) -> Dict:
    """Call POST /reset?task_id=<task_id> on the server."""
    url = f"{API_BASE_URL}/reset?task_id={urllib.parse.quote(task_id)}"
    return _http("POST", url)


def http_step(action: Action) -> Tuple[Dict, float, bool, Dict]:
    """
    Call POST /step on the server.

    Returns
    -------
    (observation_dict, reward, done, info)
    """
    resp = _http("POST", f"{API_BASE_URL}/step", body=action.model_dump())
    return resp["observation"], resp["reward"], resp["done"], resp["info"]


def http_state() -> Dict:
    """Call GET /state on the server."""
    return _http("GET", f"{API_BASE_URL}/state")


# ===========================================================================
# Observation helpers
# ===========================================================================

def observation_from_dict(d: Dict) -> Observation:
    """Parse a raw dict (from HTTP or local) into an Observation model."""
    return Observation(**d)


def state_from_dict(d: Dict) -> State:
    """Parse a raw dict into a State model."""
    return State(**d)


def _format_tasks_for_prompt(obs: Observation) -> str:
    """Render the task list into a human-readable string for the LLM prompt."""
    lines = []
    for t in obs.tasks:
        status_icon = {"not_started": "⬜", "in_progress": "🔄", "done": "✅"}.get(
            t.status, "?"
        )
        sev = f" [severity={t.severity}]" if t.severity else ""
        lines.append(
            f"  [{t.id}] {status_icon} {t.name!r}  "
            f"type={t.type}{sev}  "
            f"deadline={t.deadline}d  "
            f"remaining={t.remaining_hours}h/{t.estimated_hours}h"
        )
    return "\n".join(lines)


# ===========================================================================
# LLM client
# ===========================================================================

def _build_client():
    """
    Build an OpenAI-compatible client.

    Returns None if no API key is available (heuristic-only mode).
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning(
            "openai package not installed. Falling back to heuristic agent. "
            "Install via: pip install openai"
        )
        return None

    if not HF_TOKEN:
        logger.warning("HF_TOKEN is not set. Falling back to heuristic agent.")
        return None

    return OpenAI(base_url=LLM_BASE_URL, api_key=HF_TOKEN)


def _build_llm_prompt(task_id: str, obs: Observation) -> str:
    """
    Build a structured, JSON-requesting prompt for the current observation.

    The prompt describes the scenario context, the current state, the action
    schema, and asks the model to respond with *only* a JSON object.
    """
    cfg = TASK_CONFIGS[task_id]
    task_desc = {
        "easy_1":   "Basic task selection. 2 tasks, generous 3-day deadline.",
        "medium_1": "Time-aware prioritisation. 3 tasks, one critical bug due in 2 days.",
        "hard_1":   "Multi-day triage. 5 tasks including 2 critical bugs. Total work exceeds capacity — you must prioritise ruthlessly.",
    }.get(task_id, task_id)

    task_list = _format_tasks_for_prompt(obs)
    summary = obs.summary

    prompt = f"""You are an AI planner helping a junior developer manage their coding workload.

=== SCENARIO: {task_id} ===
{task_desc}
Max days: {cfg["max_days"]}  |  Daily capacity: {cfg["daily_capacity"]}h/day

=== CURRENT STATE ===
Day: {obs.day}
Hours left today: {obs.hours_left_today}h

Tasks:
{task_list}

Summary:
  - Tasks remaining       : {summary.tasks_remaining}
  - High-severity bugs    : {summary.high_severity_bugs_remaining}
  - Min days to deadline  : {summary.days_until_deadline}

=== YOUR GOAL ===
Choose ONE task to work on and decide how many hours to spend.
Priorities (in order):
  1. Fix HIGH-severity bugs first — especially those with deadline <= 2 days.
  2. Work on MEDIUM-severity bugs next.
  3. Then features and reviews.
  4. Use `ask_for_help: true` for tasks with estimated_hours >= 5 or high-severity bugs.
  5. Never work on a "done" task.
  6. Hours must be > 0 and <= min({obs.hours_left_today}, remaining_hours of chosen task).

=== ACTION SCHEMA ===
{{
  "task_id": <int — index of the task (0-based) to work on>,
  "hours":   <float — hours to spend, must be > 0>,
  "ask_for_help": <bool>
}}

=== RESPOND WITH ONLY THE JSON OBJECT BELOW. NO EXPLANATION. ===
"""
    return prompt.strip()


def _parse_action_from_response(text: str, obs: Observation) -> Optional[Action]:
    """
    Extract a valid Action from raw LLM response text.

    Strategy:
    1. Try to find a JSON object in the response (handles markdown code blocks).
    2. Validate the parsed dict against the Action schema.
    3. Clamp hours to allowable range if needed.

    Returns None if parsing fails after cleaning attempts.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    # Try to extract just the first {...} block
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        d = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: try the entire response as JSON
        try:
            d = json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    # Validate task_id
    n_tasks = len(obs.tasks)
    task_id_val = d.get("task_id")
    if not isinstance(task_id_val, int) or not (0 <= task_id_val < n_tasks):
        return None

    # Check task is not already done
    chosen_task = obs.tasks[task_id_val]
    if chosen_task.status == "done":
        return None

    # Clamp hours
    max_hours = min(obs.hours_left_today, chosen_task.remaining_hours)
    hours_val = float(d.get("hours", max_hours))
    hours_val = max(0.01, min(hours_val, max_hours))

    ask = bool(d.get("ask_for_help", False))

    return Action(task_id=task_id_val, hours=round(hours_val, 2), ask_for_help=ask)


# ===========================================================================
# Heuristic fallback agent
# ===========================================================================

def _heuristic_action(obs: Observation) -> Action:
    """
    Deterministic greedy fallback used when the LLM is unavailable or fails.

    Priority: high-severity bugs > near-deadline > medium bugs > features > reviews.
    Spends as many hours as possible on the chosen task.
    """
    open_tasks = [t for t in obs.tasks if t.status != "done"]
    if not open_tasks:
        # Should not happen — episode should be done
        return Action(task_id=0, hours=0.01, ask_for_help=False)

    def _priority(t):
        sev = {"high": 3, "medium": 2, "low": 1, None: 0}
        typ = {"bug": 3, "review": 2, "feature": 1}
        return (typ[t.type], sev[t.severity], -t.deadline)

    chosen = sorted(open_tasks, key=_priority, reverse=True)[0]
    hours = min(obs.hours_left_today, chosen.remaining_hours)
    ask_help = chosen.estimated_hours >= 5.0 or (
        chosen.type == "bug" and chosen.severity == "high"
    )
    return Action(task_id=chosen.id, hours=round(hours, 2), ask_for_help=ask_help)


# ===========================================================================
# LLM action selector (with retries and heuristic fallback)
# ===========================================================================

def get_action(
    client,
    task_id: str,
    obs: Observation,
    use_llm: bool = True,
) -> Action:
    """
    Request an Action from the LLM (with retries), falling back to heuristic.

    Parameters
    ----------
    client   : OpenAI client (or None for heuristic-only mode)
    task_id  : scenario identifier
    obs      : current Observation
    use_llm  : if False, always use the heuristic (--no-llm flag)

    Returns
    -------
    A valid, clamped Action.
    """
    if not use_llm or client is None:
        return _heuristic_action(obs)

    prompt = _build_llm_prompt(task_id, obs)
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw_text = response.choices[0].message.content or ""
            action = _parse_action_from_response(raw_text, obs)

            if action is not None:
                logger.debug("LLM action (attempt %d): %s", attempt, action)
                return action

            last_error = f"Failed to parse action from: {raw_text[:120]!r}"
            logger.warning("Attempt %d/%d – %s", attempt, MAX_RETRIES, last_error)

        except Exception as exc:
            last_error = str(exc)
            logger.warning("Attempt %d/%d – LLM error: %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # All retries exhausted → use heuristic
    logger.warning("All LLM retries failed (%s). Using heuristic fallback.", last_error)
    return _heuristic_action(obs)


# ===========================================================================
# Structured logging
# ===========================================================================

def log_start(task_id: str, env_url: str, model: str) -> None:
    """Emit the [START] line."""
    print(f"[START] task={task_id} env_url={env_url} model={model}", flush=True)


def log_step(
    task_id: str,
    step_num: int,
    action: Action,
    reward: float,
    done: bool,
    info: Dict,
) -> None:
    """Emit one [STEP] line."""
    action_json = json.dumps(action.model_dump(), separators=(",", ":"))
    info_json = json.dumps(
        {k: v for k, v in info.items() if not isinstance(v, float) or True},
        separators=(",", ":"),
    )
    print(
        f"[STEP] task={task_id} step={step_num} "
        f"action={action_json} "
        f"reward={round(reward, 4)} "
        f"done={done} "
        f"info={info_json}",
        flush=True,
    )


def log_end(task_id: str, score: float, steps: int) -> None:
    """Emit the [END] line."""
    print(f"[END] task={task_id} score={round(score, 4)} steps={steps}", flush=True)


# ===========================================================================
# Episode runners
# ===========================================================================

def run_episode_local(
    task_id: str,
    client,
    use_llm: bool,
    env_url: str,
) -> float:
    """
    Run one complete episode using the local Python environment.

    The LLM (or heuristic) decides actions; the environment steps locally.
    Trajectory is collected for grading.

    Returns
    -------
    float : grader score in [0.0, 1.0]
    """
    env = AssignmentPlannerEnv(task_id=task_id)
    obs = observation_from_dict(env.reset().model_dump())
    trajectory: List[State] = [env.state()]

    log_start(task_id, env_url, MODEL_NAME)

    step_num = 0
    done = False

    while not done:
        action = get_action(client, task_id, obs, use_llm=use_llm)
        raw_obs, reward, done, info = env.step(action)
        obs = observation_from_dict(raw_obs.model_dump())
        trajectory.append(env.state())

        step_num += 1
        log_step(task_id, step_num, action, reward, done, info)

    score = grade(task_id, trajectory)
    log_end(task_id, score, step_num)
    return score


def run_episode_http(
    task_id: str,
    client,
    use_llm: bool,
    env_url: str,
) -> float:
    """
    Run one complete episode against the HTTP server.

    POST /reset to initialise, then POST /step in a loop.
    A local env shadow is kept in sync via env.state() mirroring so we can
    call the graders (which need a List[State] trajectory) at the end.

    Returns
    -------
    float : grader score in [0.0, 1.0]
    """
    # Reset
    try:
        raw_obs_dict = http_reset(task_id)
    except RuntimeError as exc:
        logger.error("Failed to reset via HTTP: %s", exc)
        raise

    obs = observation_from_dict(raw_obs_dict)

    # Keep a local shadow env for grading (graders need State objects)
    shadow_env = AssignmentPlannerEnv(task_id=task_id)
    shadow_env.reset()
    trajectory: List[State] = [shadow_env.state()]

    log_start(task_id, env_url, MODEL_NAME)

    step_num = 0
    done = False

    while not done:
        action = get_action(client, task_id, obs, use_llm=use_llm)

        try:
            raw_obs_dict, reward, done, info = http_step(action)
        except RuntimeError as exc:
            logger.error("HTTP /step failed: %s. Using heuristic & local env.", exc)
            # Fallback: step in shadow env
            raw_obs_pydantic, reward, done, info = shadow_env.step(action)
            raw_obs_dict = raw_obs_pydantic.model_dump()

        obs = observation_from_dict(raw_obs_dict)

        # Mirror the action in the shadow env to keep trajectory in sync
        try:
            shadow_env.step(action)
        except Exception:
            pass  # shadow may diverge on clamp edge-cases; grader still runs
        trajectory.append(shadow_env.state())

        step_num += 1
        log_step(task_id, step_num, action, reward, done, info)

    score = grade(task_id, trajectory)
    log_end(task_id, score, step_num)
    return score


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline inference script for the Assignment Planner Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=USE_LOCAL_ENV,
        help="Use local AssignmentPlannerEnv instead of HTTP server.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Use the heuristic greedy agent instead of the LLM (for quick tests).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list_task_ids(),
        metavar="TASK_ID",
        help=f"Explicit task IDs to run. Choices: {list_task_ids()}",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Pick N random tasks from each difficulty section (easy/medium/hard). "
            "E.g. --random 1 picks 1 easy + 1 medium + 1 hard = 3 tasks total. "
            "--random 5 picks 5 from each section = 15 tasks total. "
            "Ignored if --tasks is also specified."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for --random sampling (omit for a different set each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_llm = not args.no_llm
    use_local = args.local

    # ── Resolve task list ────────────────────────────────────────────────────
    if args.tasks is not None:
        # User explicitly named tasks → use those
        tasks_to_run = args.tasks
    elif args.random and args.random > 0:
        # Random sampling mode: pick N from each difficulty section
        tasks_to_run = sample_tasks(n=args.random, seed=args.seed)
    else:
        # Default: run the canonical 3 tasks (easy_1, medium_1, hard_1)
        tasks_to_run = ["easy_1", "medium_1", "hard_1"]

    logger.info("=" * 60)
    logger.info("Assignment & Bug-Fix Planner Agent – Baseline Inference")
    logger.info("=" * 60)
    logger.info("Mode       : %s", "local env" if use_local else "HTTP server")
    logger.info("LLM        : %s", MODEL_NAME if use_llm else "heuristic (no LLM)")
    logger.info("Server URL : %s", API_BASE_URL)
    logger.info("Tasks      : %s", tasks_to_run)
    logger.info("=" * 60)

    # Build LLM client (None if unavailable)
    client = _build_client() if use_llm else None

    # Environment URL shown in [START] lines
    env_url = "local" if use_local else API_BASE_URL

    all_scores: Dict[str, float] = {}
    total_start = time.time()

    for task_id in tasks_to_run:
        logger.info("Running task: %s", task_id)
        t0 = time.time()

        try:
            if use_local:
                score = run_episode_local(task_id, client, use_llm, env_url)
            else:
                score = run_episode_http(task_id, client, use_llm, env_url)
        except Exception as exc:
            logger.error("Episode failed for %s: %s", task_id, exc)
            score = 0.0
            log_end(task_id, score, 0)

        elapsed = time.time() - t0
        all_scores[task_id] = score
        logger.info("Task %s finished in %.1fs  score=%.4f", task_id, elapsed, score)

    # ── Final summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 50)
    print("FINAL SCORES")
    print("=" * 50)
    for tid, sc in all_scores.items():
        bar = "=" * int(sc * 20)
        print(f"  {tid:<12}  score={sc:.4f}  |{bar:<20}|")
    mean = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"\n  Mean score  : {mean:.4f}")
    print(f"  Total time  : {total_elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
