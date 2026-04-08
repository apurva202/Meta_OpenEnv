from __future__ import annotations

import sys
import logging
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global IO and Logging Initialization
# ---------------------------------------------------------------------------
# Ensure stdout is unbuffered and uses UTF-8 to satisfy the validator
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# Direct all standard logging to stderr so it doesn't pollute the validator's stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.envs.assignment_planner.environment import AssignmentPlannerEnv
from src.envs.assignment_planner.graders import grade
from src.envs.assignment_planner.models import Action, Observation, State
from src.envs.assignment_planner.task_config import TASK_CONFIGS, list_task_ids, sample_tasks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# API_BASE_URL is the LLM proxy URL injected by the platform
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
# Platform injects API_KEY; fall back to HF_TOKEN for local testing
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
USE_LOCAL_ENV: bool = os.getenv("USE_LOCAL_ENV", "0") == "1"

MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
MAX_TOKENS: int = 256
TEMPERATURE: float = 0.2

# ===========================================================================
# HTTP client helpers
# ===========================================================================

def _http(method: str, url: str, body: Optional[Dict] = None, timeout: int = 30) -> Dict:
    data = json.dumps(body).encode() if body is not None else None
    headers: Dict[str, str] = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc

def http_reset(task_id: str) -> Dict:
    url = f"{API_BASE_URL}/reset?task_id={urllib.parse.quote(task_id)}"
    return _http("POST", url)

def http_step(action: Action) -> Tuple[Dict, float, bool, Dict]:
    resp = _http("POST", f"{API_BASE_URL}/step", body=action.model_dump())
    return resp["observation"], resp["reward"], resp["done"], resp["info"]

# ===========================================================================
# Helper / LLM Logic
# ===========================================================================

def observation_from_dict(d: Dict) -> Observation:
    return Observation(**d)

def _build_client():
    """Build OpenAI client using platform-injected API_BASE_URL and API_KEY."""
    try:
        from openai import OpenAI
        if not API_KEY:
            logger.warning("No API_KEY or HF_TOKEN found. Falling back to heuristic agent.")
            return None
        logger.info("LLM client initialized: %s", API_BASE_URL)
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except ImportError:
        logger.warning("openai package not installed. Falling back to heuristic agent.")
        return None

def _heuristic_action(obs: Observation) -> Action:
    open_tasks = [t for t in obs.tasks if t.status != "done"]
    if not open_tasks: return Action(task_id=0, hours=0.01, ask_for_help=False)
    chosen = sorted(open_tasks, key=lambda t: (t.type == "bug", t.severity == "high", -t.deadline), reverse=True)[0]
    hours = min(obs.hours_left_today, chosen.remaining_hours)
    return Action(task_id=chosen.id, hours=round(hours, 2), ask_for_help=(chosen.severity == "high"))

# ===========================================================================
# LLM action selector
# ===========================================================================

def _build_llm_prompt(task_id: str, obs: Observation) -> str:
    cfg = TASK_CONFIGS[task_id]
    task_desc = {
        "easy_1": "Basic task selection. 2 tasks.",
        "medium_1": "Time-aware prioritisation. 3 tasks.",
        "hard_1": "Multi-day triage. 5 tasks. ruthlessly prioritise.",
    }.get(task_id, task_id)

    task_list = "\n".join([f" [{t.id}] {t.name} (type={t.type}, sev={t.severity}, deadline={t.deadline}d, rem={t.remaining_hours}h)" for t in obs.tasks])
    
    return f"""You are an AI planner.
Scenario: {task_id} - {task_desc}
Capacity: {cfg["daily_capacity"]}h/day, Day: {obs.day}, Left: {obs.hours_left_today}h
Tasks:
{task_list}
Goal: Pick ONE task. Respond with ONLY JSON: {{"task_id": int, "hours": float, "ask_for_help": bool}}"""

def _parse_action_from_response(text: str, obs: Observation) -> Optional[Action]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: return None
        d = json.loads(match.group(0))
        tid = d.get("task_id")
        if not (isinstance(tid, int) and 0 <= tid < len(obs.tasks)): return None
        if obs.tasks[tid].status == "done": return None
        
        max_h = min(obs.hours_left_today, obs.tasks[tid].remaining_hours)
        h = max(0.01, min(float(d.get("hours", max_h)), max_h))
        return Action(task_id=tid, hours=round(h, 2), ask_for_help=bool(d.get("ask_for_help", False)))
    except: return None

def get_action(client, task_id: str, obs: Observation, use_llm: bool = True) -> Action:
    if not use_llm or client is None:
        return _heuristic_action(obs)

    prompt = _build_llm_prompt(task_id, obs)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            action = _parse_action_from_response(resp.choices[0].message.content, obs)
            if action: return action
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)

    return _heuristic_action(obs)

# ===========================================================================
# STRUCTURED LOGGING (The Fix for Phase 2)
# ===========================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ===========================================================================
# Episode runners
# ===========================================================================

def run_episode(task_id: str, client, use_llm: bool, use_local: bool) -> float:
    env_url = "local" if use_local else API_BASE_URL
    log_start(task_id, env_url, MODEL_NAME)
    
    if use_local:
        env = AssignmentPlannerEnv(task_id=task_id)
        obs = observation_from_dict(env.reset().model_dump())
        shadow_env = env
    else:
        obs = observation_from_dict(http_reset(task_id))
        shadow_env = AssignmentPlannerEnv(task_id=task_id)
        shadow_env.reset()

    trajectory: List[State] = [shadow_env.state()]
    rewards_list: List[float] = []
    step_num = 0
    done = False

    while not done:
        action = get_action(client, task_id, obs, use_llm)
        
        if use_local:
            raw_obs, reward, done, _ = env.step(action)
            obs = observation_from_dict(raw_obs.model_dump())
        else:
            raw_obs_dict, reward, done, _ = http_step(action)
            obs = observation_from_dict(raw_obs_dict)
            try: shadow_env.step(action)
            except: pass
        
        trajectory.append(shadow_env.state())
        rewards_list.append(reward)
        step_num += 1
        # Convert Action object to string for the log, matching the sample
        action_str = f'work(task={action.task_id}, hours={action.hours})'
        log_step(step_num, action_str, reward, done)

    score = grade(task_id, trajectory)
    success = score >= 0.1
    log_end(success, step_num, score, rewards_list)
    return score

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=USE_LOCAL_ENV)
    parser.add_argument("--no-llm", action="store_true", default=False)
    args = parser.parse_args()

    # Use the stable baseline tasks for evaluation
    tasks_to_run = ["easy_1", "medium_1", "hard_1"]

    client = _build_client() if not args.no_llm else None

    for task_id in tasks_to_run:
        try:
            run_episode(task_id, client, not args.no_llm, args.local)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")

if __name__ == "__main__":
    main()
