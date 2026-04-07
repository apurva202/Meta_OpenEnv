"""
app.py
------
OpenEnv-compliant FastAPI server for the Assignment & Bug-Fix Planner Agent.

Exposes three endpoints:
  POST /reset  – initialise (or re-initialise) the environment for a given task_id.
  POST /step   – advance the environment by one agent action.
  GET  /state  – retrieve the full internal state (debugging / logging).

A single global ``AssignmentPlannerEnv`` instance is used per container.
This matches the OpenEnv single-agent judging model where one episode runs
at a time.  For multi-tenant usage, replace with a session-keyed dict.

Environment variables
---------------------
PORT          : TCP port to bind (default 7860, set by HF Spaces).
DEFAULT_TASK  : task_id used when /reset is called without a task_id param
                (default "easy_1").
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path bootstrap - allow imports from src/envs
# ---------------------------------------------------------------------------
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Add src and src/envs to PYTHONPATH so it finds assignment_planner natively
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src", "envs")))

from assignment_planner.environment import AssignmentPlannerEnv
from assignment_planner.models import Action, Observation, State
from assignment_planner.task_config import list_task_ids

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("assignment_planner.server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TASK: str = os.getenv("DEFAULT_TASK", "easy_1")
PORT: int = int(os.getenv("PORT", "7860"))

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
# Initialised during lifespan startup so FastAPI can report errors before
# accepting any traffic.

_env: Optional[AssignmentPlannerEnv] = None


def _get_env() -> AssignmentPlannerEnv:
    """Return the global env, raising 503 if not yet initialised."""
    if _env is None:
        raise HTTPException(
            status_code=503,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the global env on startup with the default task."""
    global _env
    logger.info("Server starting – initialising env with task_id='%s'", DEFAULT_TASK)
    try:
        _env = AssignmentPlannerEnv(task_id=DEFAULT_TASK)
        logger.info("Environment ready. Available task IDs: %s", list_task_ids())
    except Exception as exc:
        logger.error("Failed to initialise environment: %s", exc)
        raise
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Assignment & Bug-Fix Planner Agent",
    description=(
        "OpenEnv-compliant HTTP server wrapping the AssignmentPlannerEnv. "
        "An AI agent decides which coding tasks to work on, how many hours "
        "to spend, and whether to ask for help, across multiple days."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow all origins so HF Space iframe / external agents can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """Structured response returned by POST /step."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

    model_config = {"json_schema_extra": {
        "example": {
            "observation": {"day": 0, "hours_left_today": 3.0, "tasks": [], "summary": {}},
            "reward": 1.75,
            "done": False,
            "info": {"clamped_hours": 3.0, "day_advanced": False},
        }
    }}


class ErrorResponse(BaseModel):
    """Standard JSON error body."""

    detail: str


class HealthResponse(BaseModel):
    status: str
    current_task_id: Optional[str]
    available_tasks: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    tags=["meta"],
)
async def root() -> HealthResponse:
    """
    Health-check endpoint.  Returns server status and currently loaded task_id.
    """
    current = _env.task_id if _env is not None else None
    return HealthResponse(
        status="ok",
        current_task_id=current,
        available_tasks=list_task_ids(),
    )


@app.post(
    "/reset",
    response_model=Observation,
    summary="Reset the environment",
    tags=["env"],
    responses={400: {"model": ErrorResponse}},
)
async def reset(
    task_id: str = Query(
        default=DEFAULT_TASK,
        description=(
            "Scenario to load. One of: "
            + ", ".join(f"'{t}'" for t in list_task_ids())
        ),
    ),
) -> Observation:
    """
    Reset (or re-initialise) the environment for the given ``task_id``.

    - Replaces the global env instance with a fresh one.
    - Returns the initial ``Observation``.

    **Example**
    ```
    POST /reset?task_id=medium_1
    ```
    """
    global _env

    valid_ids = list_task_ids()
    if task_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {valid_ids}"
            ),
        )

    try:
        _env = AssignmentPlannerEnv(task_id=task_id)
        obs = _env.reset()
        logger.info("Environment reset with task_id='%s'", task_id)
        return obs
    except Exception as exc:
        logger.exception("Error during reset: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/step",
    response_model=StepResponse,
    summary="Advance the environment by one action",
    tags=["env"],
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def step(action: Action) -> StepResponse:
    """
    Apply one agent **Action** to the environment and return the result.

    **Request body** (JSON):
    ```json
    {
      "task_id": 0,
      "hours": 2.0,
      "ask_for_help": false
    }
    ```

    **Fields**
    - ``task_id``      – 0-based index into the current task list.
    - ``hours``        – Hours to work; must be > 0.  Clamped internally to
                         ``min(hours_left_today, remaining_hours[task_id])``.
    - ``ask_for_help`` – Whether to ask a mentor for help this step.

    **Returns** ``StepResponse`` with fields::
        observation  – next Observation
        reward       – dense shaped reward (float)
        done         – True when episode has ended
        info         – auxiliary diagnostic dict
    """
    env = _get_env()

    try:
        obs, reward, done, info = env.step(action)
        logger.debug(
            "step | task=%d hours=%.1f reward=%.4f done=%s",
            action.task_id, action.hours, reward, done,
        )
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except AssertionError as exc:
        # Raised by env.step() on invalid actions
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /step: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/state",
    response_model=State,
    summary="Get full internal state",
    tags=["env"],
    responses={503: {"model": ErrorResponse}},
)
async def state() -> State:
    """
    Return the **full internal state** of the environment.

    Useful for debugging, logging, or trajectory recording.
    Does not advance the episode.

    **Returns** ``State`` with fields::
        day              – current day index (0-based)
        hours_left_today – remaining work hours today
        tasks            – complete task list with up-to-date status
    """
    env = _get_env()
    return env.state()


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError):
    return JSONResponse(status_code=400, content={"detail": f"Key error: {exc}"})


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
