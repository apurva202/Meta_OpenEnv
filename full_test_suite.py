"""
full_test_suite.py
==================
Comprehensive end-to-end tests for all four phases of the
Assignment & Bug-Fix Planner Agent project.

Tests:
  1. Pydantic model validation
  2. task_config.py
  3. AssignmentPlannerEnv: reset(), step(), state()
  4. Day-boundary and deadline logic
  5. Graders: grade_easy, grade_medium, grade_hard, grade()
  6. FastAPI server: /reset, /step, /state, / (health)
  7. inference.py: --local --no-llm (all 3 tasks)

Run:  python full_test_suite.py
"""

from __future__ import annotations

import json
import sys
import time
import threading
import urllib.request
import urllib.error
import urllib.parse
import traceback
from typing import List, Dict, Any

# ── path bootstrap ────────────────────────────────────────────────────────────
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results: List[Dict[str, Any]] = []

def section(title: str):
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}")

def ok(msg: str):
    results.append({"pass": True, "msg": msg})
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def fail(msg: str, exc: Exception | None = None):
    results.append({"pass": False, "msg": msg})
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if exc:
        print(f"         {RED}{exc}{RESET}")

def expect_equal(a, b, label: str):
    if a == b:
        ok(f"{label}: {a!r}")
    else:
        fail(f"{label}: expected {b!r}, got {a!r}")

def expect_true(cond: bool, label: str):
    if cond:
        ok(label)
    else:
        fail(label)


# ═══════════════════════════════════════════════════════════════
# TEST 1 – Pydantic Models
# ═══════════════════════════════════════════════════════════════
section("TEST 1 – Pydantic Models")

from pydantic import ValidationError
from src.envs.assignment_planner.models import Task, Action, Observation, State, Summary

try:
    t = Task(id=0, name="Fix login bug", type="bug", severity="high",
             deadline=2, estimated_hours=3.0, remaining_hours=3.0, status="not_started")
    ok("Task model: valid construction")
except Exception as e:
    fail("Task model: valid construction", e)

try:
    Action(task_id=0, hours=0.0, ask_for_help=False)
    fail("Action model: should reject hours=0")
except ValidationError:
    ok("Action model: hours=0 correctly rejected (pydantic gt constraint)")

try:
    a = Action(task_id=1, hours=2.5, ask_for_help=True)
    expect_equal(a.hours, 2.5, "Action.hours")
    expect_equal(a.ask_for_help, True, "Action.ask_for_help")
except Exception as e:
    fail("Action model: valid construction", e)

try:
    s = Summary(tasks_remaining=3, high_severity_bugs_remaining=1, days_until_deadline=2)
    obs = Observation(day=0, hours_left_today=6.0, tasks=[t], summary=s)
    obs2 = Observation(**obs.model_dump())
    expect_equal(obs2.day, obs.day, "Observation round-trip day")
    expect_equal(len(obs2.tasks), 1, "Observation round-trip tasks len")
except Exception as e:
    fail("Observation round-trip", e)

try:
    st = State(day=1, hours_left_today=4.0, tasks=[t])
    ok(f"State model: day={st.day} hours={st.hours_left_today}")
except Exception as e:
    fail("State model", e)


# ═══════════════════════════════════════════════════════════════
# TEST 2 – task_config.py
# ═══════════════════════════════════════════════════════════════
section("TEST 2 – task_config.py")

from src.envs.assignment_planner.task_config import list_task_ids, get_config, TASK_CONFIGS

ids = list_task_ids()
expect_equal(ids, ["easy_1", "medium_1", "hard_1"], "list_task_ids()")

expected_tasks  = {"easy_1": 2, "medium_1": 3, "hard_1": 5}
expected_days   = {"easy_1": 3, "medium_1": 4, "hard_1": 3}
expected_cap    = {"easy_1": 6.0, "medium_1": 6.0, "hard_1": 6.0}

for tid in ids:
    cfg = get_config(tid)
    expect_equal(len(cfg["tasks"]),      expected_tasks[tid],  f"{tid} task count")
    expect_equal(cfg["max_days"],        expected_days[tid],   f"{tid} max_days")
    expect_equal(cfg["daily_capacity"],  expected_cap[tid],    f"{tid} daily_capacity")

try:
    get_config("does_not_exist")
    fail("get_config(unknown): should raise KeyError")
except KeyError:
    ok("get_config(unknown): raises KeyError")


# ═══════════════════════════════════════════════════════════════
# TEST 3 – AssignmentPlannerEnv: reset()
# ═══════════════════════════════════════════════════════════════
section("TEST 3 – AssignmentPlannerEnv reset()")

from src.envs.assignment_planner.environment import AssignmentPlannerEnv

for tid in ids:
    try:
        env = AssignmentPlannerEnv(task_id=tid)
        obs = env.reset()
        cfg = get_config(tid)
        expect_equal(obs.day, 0, f"{tid} day after reset")
        expect_equal(obs.hours_left_today, cfg["daily_capacity"], f"{tid} hours_left_today")
        expect_equal(len(obs.tasks), len(cfg["tasks"]), f"{tid} task count")
        for t in obs.tasks:
            expect_equal(t.status, "not_started", f"{tid} task[{t.id}].status")
            expect_equal(t.remaining_hours, t.estimated_hours,
                         f"{tid} task[{t.id}].remaining_hours")
    except Exception as e:
        fail(f"{tid} reset()", e)

try:
    AssignmentPlannerEnv(task_id="bad_id")
    fail("Unknown task_id should raise KeyError")
except KeyError:
    ok("Unknown task_id raises KeyError")


# ═══════════════════════════════════════════════════════════════
# TEST 4 – Environment step() + state()
# ═══════════════════════════════════════════════════════════════
section("TEST 4 – Environment step() and state()")

# easy_1: 2 tasks (2h bug + 4h feature), 6h/day cap
env = AssignmentPlannerEnv(task_id="easy_1")
env.reset()

# Step 1: finish the bug (2h)
obs2, r1, done1, info1 = env.step(Action(task_id=0, hours=2.0, ask_for_help=True))
expect_true(not done1,               "easy_1 step1: not done yet")
expect_equal(obs2.tasks[0].status,   "done",  "easy_1 step1: task0 done")
expect_equal(obs2.hours_left_today,  4.0,     "easy_1 step1: 4h left")
expect_true(r1 > 0,                  f"easy_1 step1: reward positive ({r1:.4f})")
expect_equal(info1["clamped_hours"], 2.0,     "easy_1 step1: clamped_hours")

# Step 2: finish feature (4h) — day boundary crossed, all done
obs3, r2, done2, info2 = env.step(Action(task_id=1, hours=4.0, ask_for_help=False))
expect_true(done2,                   "easy_1 step2: episode done")
expect_true(info2["all_tasks_done"], "easy_1 step2: all_tasks_done=True")
expect_true(info2["day_advanced"],   "easy_1 step2: day advanced")
expect_true(r2 > 0,                  f"easy_1 step2: reward positive ({r2:.4f})")

# state() reflects done
st = env.state()
expect_true(all(t.status == "done" for t in st.tasks),
            "easy_1 state(): all tasks done")

# Test day advancement + deadline decrement
env_m = AssignmentPlannerEnv(task_id="medium_1")
env_m.reset()
# Exhaust day 0: spend 6h on task 0 (3h bug → done, then 3h from task 1)
obs_m1, _, _, info_m1 = env_m.step(Action(task_id=0, hours=3.0, ask_for_help=False))
# Should NOT have advanced day since 3h used, but 3h remain
expect_true(not info_m1["day_advanced"], "medium_1: day not advanced after 3h")
expect_equal(obs_m1.tasks[0].status, "done", "medium_1: task0 done after 3h")

obs_m2, _, _, info_m2 = env_m.step(Action(task_id=1, hours=3.0, ask_for_help=False))
expect_true(info_m2["day_advanced"], "medium_1: day advanced when capacity exhausted")
expect_equal(obs_m2.day, 1, "medium_1: day becomes 1")


# ═══════════════════════════════════════════════════════════════
# TEST 5 – Invalid action handling
# ═══════════════════════════════════════════════════════════════
section("TEST 5 – Invalid action handling")

env3 = AssignmentPlannerEnv(task_id="medium_1")
env3.reset()

# Out-of-range task_id
try:
    env3.step(Action(task_id=99, hours=1.0, ask_for_help=False))
    fail("task_id=99 should raise AssertionError")
except AssertionError:
    ok("task_id out-of-range raises AssertionError")

# Already-done task (complete task 0 first, then try again)
env4 = AssignmentPlannerEnv(task_id="easy_1")
env4.reset()
env4.step(Action(task_id=0, hours=2.0, ask_for_help=False))
try:
    env4.step(Action(task_id=0, hours=1.0, ask_for_help=False))
    fail("Stepping on done task should raise AssertionError")
except AssertionError:
    ok("Stepping on done task raises AssertionError")

# Excessive hours get clamped, not errored
env5 = AssignmentPlannerEnv(task_id="medium_1")
env5.reset()
_, _, _, info5 = env5.step(Action(task_id=0, hours=999.0, ask_for_help=False))
expect_true(info5["clamped_hours"] <= 6.0,
            f"Excessive hours clamped to {info5['clamped_hours']}h")


# ═══════════════════════════════════════════════════════════════
# TEST 6 – Graders
# ═══════════════════════════════════════════════════════════════
section("TEST 6 – Graders")

from src.envs.assignment_planner.graders import (
    grade_easy, grade_medium, grade_hard, grade, GRADER_MAP
)

def run_full_heuristic_episode(task_id: str):
    """Run a greedy heuristic episode and return the trajectory."""
    from src.envs.assignment_planner.models import Action as Act

    def priority(t):
        sev = {"high": 3, "medium": 2, "low": 1, None: 0}
        typ = {"bug": 3, "review": 2, "feature": 1}
        return (typ[t.type], sev[t.severity], -t.deadline)

    env = AssignmentPlannerEnv(task_id=task_id)
    env.reset()
    trajectory = [env.state()]
    done = False
    while not done:
        open_t = [t for t in env._tasks if t.status != "done"]
        if not open_t:
            break
        chosen = sorted(open_t, key=priority, reverse=True)[0]
        h = min(env._hours_left_today, chosen.remaining_hours)
        _, _, done, _ = env.step(Act(task_id=chosen.id, hours=h, ask_for_help=False))
        trajectory.append(env.state())
    return trajectory

for tid in ids:
    traj = run_full_heuristic_episode(tid)
    s = grade(tid, traj)
    expect_true(0.0 <= s <= 1.0, f"{tid}: score in [0,1] ({s:.4f})")

# Determinism: same trajectory → same score
traj_e = run_full_heuristic_episode("easy_1")
s1 = grade_easy(traj_e)
s2 = grade_easy(traj_e)
expect_equal(s1, s2, "grade_easy is deterministic")

# Empty trajectory → 0
expect_equal(grade_easy([]), 0.0, "grade_easy([]) = 0.0")

# Dispatching
try:
    grade("bad_task", [])
    fail("grade(unknown) should raise KeyError")
except KeyError:
    ok("grade(unknown) raises KeyError")

# Coefficient expectations (heuristic greedy should score high)
s_easy   = grade("easy_1",   run_full_heuristic_episode("easy_1"))
s_medium = grade("medium_1", run_full_heuristic_episode("medium_1"))
s_hard   = grade("hard_1",   run_full_heuristic_episode("hard_1"))
print(f"\n  Heuristic scores: easy={s_easy:.4f}  medium={s_medium:.4f}  hard={s_hard:.4f}")
expect_true(s_easy   >= 0.9, f"easy_1 heuristic score >= 0.9 ({s_easy:.4f})")
expect_true(s_medium >= 0.9, f"medium_1 heuristic score >= 0.9 ({s_medium:.4f})")
expect_true(s_hard   >= 0.5, f"hard_1 heuristic score >= 0.5 ({s_hard:.4f})")


# ═══════════════════════════════════════════════════════════════
# TEST 7 – FastAPI Server (live HTTP)
# ═══════════════════════════════════════════════════════════════
section("TEST 7 – FastAPI Server (live HTTP endpoints)")

# Build the module alias table the server's path bootstrap needs
import src.envs.assignment_planner as _pkg
sys.modules.setdefault("assignment_planner", _pkg)
import src.envs.assignment_planner.models as _m
sys.modules.setdefault("assignment_planner.models", _m)
import src.envs.assignment_planner.task_config as _tc
sys.modules.setdefault("assignment_planner.task_config", _tc)
import src.envs.assignment_planner.environment as _e
sys.modules.setdefault("assignment_planner.environment", _e)
import src.envs.assignment_planner.graders as _gr
sys.modules.setdefault("assignment_planner.graders", _gr)

sys.path.insert(0, os.path.join(_ROOT, "server"))
import app as server_app
import uvicorn

PORT_TEST = 18765

def _start_server():
    uvicorn.run(server_app.app, host="127.0.0.1", port=PORT_TEST, log_level="error")

srv = threading.Thread(target=_start_server, daemon=True)
srv.start()
time.sleep(2.5)   # wait for startup

BASE = f"http://127.0.0.1:{PORT_TEST}"

def http(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

# GET /  health check
try:
    h = http("GET", "/")
    expect_equal(h["status"], "ok", "GET /: status=ok")
    expect_equal(sorted(h["available_tasks"]), sorted(ids), "GET /: available_tasks")
except Exception as e:
    fail("GET /", e)

# POST /reset — each task
for tid in ids:
    try:
        obs = http("POST", f"/reset?task_id={tid}")
        expect_equal(obs["day"], 0, f"POST /reset?task_id={tid}: day=0")
        expect_equal(len(obs["tasks"]), expected_tasks[tid],
                     f"POST /reset?task_id={tid}: task count")
    except Exception as e:
        fail(f"POST /reset?task_id={tid}", e)

# POST /step — after resetting to easy_1
try:
    http("POST", "/reset?task_id=easy_1")
    step_resp = http("POST", "/step", {"task_id": 0, "hours": 2.0, "ask_for_help": True})
    expect_true("observation" in step_resp, "POST /step: has 'observation'")
    expect_true("reward" in step_resp,      "POST /step: has 'reward'")
    expect_true("done" in step_resp,        "POST /step: has 'done'")
    expect_true("info" in step_resp,        "POST /step: has 'info'")
    expect_equal(step_resp["observation"]["tasks"][0]["status"], "done",
                 "POST /step: task0 done after 2h")
    expect_true(step_resp["reward"] > 0,    f"POST /step: reward > 0 ({step_resp['reward']})")
except Exception as e:
    fail("POST /step", e)

# GET /state
try:
    st = http("GET", "/state")
    expect_true("day" in st and "hours_left_today" in st and "tasks" in st,
                "GET /state: has all fields")
except Exception as e:
    fail("GET /state", e)

# 400 on unknown task_id
try:
    http("POST", "/reset?task_id=nonexistent")
    fail("POST /reset?task_id=nonexistent: should return 400")
except urllib.error.HTTPError as e:
    expect_equal(e.code, 400, "POST /reset?task_id=nonexistent: returns HTTP 400")
except Exception as e:
    fail("POST /reset?task_id=nonexistent error handling", e)

# 400 on bad action
try:
    http("POST", "/reset?task_id=easy_1")
    http("POST", "/step", {"task_id": 99, "hours": 1.0, "ask_for_help": False})
    fail("POST /step bad task_id: should return 400")
except urllib.error.HTTPError as e:
    expect_equal(e.code, 400, "POST /step bad task_id: returns HTTP 400")
except Exception as e:
    fail("POST /step bad task_id error", e)


# ═══════════════════════════════════════════════════════════════
# TEST 8 – inference.py (--local --no-llm)
# ═══════════════════════════════════════════════════════════════
section("TEST 8 – inference.py (--local --no-llm, all tasks)")

import subprocess
result = subprocess.run(
    [sys.executable, "inference.py", "--local", "--no-llm"],
    capture_output=True, text=True, timeout=60, cwd=_ROOT
)

stdout = result.stdout + result.stderr
print("\n  --- inference.py output (last 20 lines) ---")
for line in stdout.strip().split("\n")[-20:]:
    print(f"  {line}")
print("  ---")

expect_equal(result.returncode, 0, "inference.py exit code = 0")

# Check structured log lines
start_lines = [l for l in stdout.split("\n") if l.startswith("[START]")]
step_lines  = [l for l in stdout.split("\n") if l.startswith("[STEP]")]
end_lines   = [l for l in stdout.split("\n") if l.startswith("[END]")]

expect_equal(len(start_lines), 3, "[START] lines count = 3")
expect_true(len(step_lines) >= 3, f"[STEP] lines >= 3 (got {len(step_lines)})")
expect_equal(len(end_lines),  3, "[END] lines count = 3")

# Validate [END] scores are in [0,1]
for line in end_lines:
    try:
        score_str = line.split("score=")[1].strip()
        score_val = float(score_str)
        expect_true(0.0 <= score_val <= 1.0,
                    f"[END] score in [0,1]: {score_val:.4f}")
    except Exception as ex:
        fail(f"Could not parse score from: {line!r}", ex)

# Validate [STEP] lines parse correctly
for line in step_lines[:3]:
    try:
        parts = dict(p.split("=", 1) for p in line[len("[STEP] "):].split(", ", 4))
        _ = int(parts["step"])
        _ = json.loads(parts["action"])
        _ = float(parts["reward"])
        _ = parts["done"] in ("True", "False")
        ok(f"[STEP] line parses OK: step={parts['step']}")
    except Exception as ex:
        fail(f"[STEP] line parse failed: {line[:80]!r}", ex)
    break  # test just the first STEP line format


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
passed = sum(1 for r in results if r["pass"])
failed = sum(1 for r in results if not r["pass"])
total  = len(results)

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  TEST SUMMARY{RESET}")
print(f"{BOLD}{'='*60}{RESET}")
print(f"  {GREEN}Passed : {passed}/{total}{RESET}")
if failed:
    print(f"  {RED}Failed : {failed}/{total}{RESET}")
    print(f"\n  {RED}Failed tests:{RESET}")
    for r in results:
        if not r["pass"]:
            print(f"    {RED}X {r['msg']}{RESET}")

print(f"\n  Mean heuristic score : easy={s_easy:.4f}  medium={s_medium:.4f}  hard={s_hard:.4f}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if failed == 0 else 1)


# So what we have done ghere is that how we got here so that's the main catch here , so what we have got here is that we have  have completed the pydantic modules but still we got a do a lot of things , so let's start with the task config module 
# in config modules we started this with a lot of things we  still got a lot of things which we still need to figure out but yeah but we still have got a long way but yeah after all it dosent matter
# the HTTp server has still got several probkems of its own but it mainly could start on its own , but still we can figure out a lot of things
#  