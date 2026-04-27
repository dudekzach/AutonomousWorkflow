from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import time
import traceback
import uuid
import threading
import os
import json
import redis

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autonomous_compare_runner import run_autonomous_compare


app = FastAPI(title="Autonomous Workflow API")

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "index.html"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# TEMPORARY DEBUG SWITCH:
# Set to True to force optimizer off for all API calls.
# Set back to False once you've finished diagnosing the 502 issue.
FORCE_OPTIMIZER_OFF = True

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("Missing REDIS_URL environment variable")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class RunRequest(BaseModel):
    prompt: str
    max_iterations: int = 3
    user_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


STAGE_PROGRESS = {
    "queued": 0,
    "starting": 5,
    "optimizing_prompt": 15,
    "running_models": 40,
    "judging": 65,
    "follow_up": 78,
    "stitching": 88,
    "saving_artifacts": 95,
    "completed": 100,
    "failed": 100,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_job_key(job_id: str) -> str:
    return f"job:{job_id}"


def save_job(job_id: str, job_data: Dict[str, Any]) -> None:
    redis_client.set(get_job_key(job_id), json.dumps(job_data))


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    raw = redis_client.get(get_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw)


def build_step(status: str = "pending") -> Dict[str, Any]:
    return {
        "status": status,
        "started_at": None,
        "completed_at": None,
        "latency_seconds": None,
        "message": None,
        "error": None,
        "provider": None,
        "model": None,
        "attempts": 0,
        "meta": {},
    }


def build_job_document(
    job_id: str,
    prompt: str,
    max_iterations: int,
    user_id: Optional[str],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    opts = dict(options or {})
    attached_filenames = opts.get("attached_filenames", [])

    timestamp = now_iso()

    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "completed_at": None,
        "request": {
            "prompt": prompt,
            "max_iterations": max_iterations,
            "user_id": user_id,
            "options": opts,
            "attached_filenames": attached_filenames,
        },
        "progress": {
            "current_step": "queued",
            "percent": 0,
            "message": "Job queued",
            "steps_completed": 0,
            "total_steps": 7,
        },
        "steps": {
            "optimizer": build_step(),
            "openai": build_step(),
            "claude": build_step(),
            "judge": build_step(),
            "follow_up": build_step(),
            "stitch": build_step(),
            "artifacts": build_step(),
        },
        "iteration": {
            "current": 0,
            "max": max_iterations,
            "history": [],
        },
        "summary": {
            "winner": None,
            "next_action": None,
            "confidence": None,
            "reason": None,
        },
        "outputs": {
            "optimized_prompt": None,
            "openai_output": None,
            "claude_output": None,
            "judge_output": None,
            "follow_up_output": None,
            "stitched_output": None,
            "final_output": None,
        },
        "artifacts": {
            "html_path": None,
            "json_path": None,
        },
        "timing": {
            "total_runtime_seconds": None,
        },
        "logs": [],
        "error": None,
        "errors": [],
        "display": {
            "status_label": "Queued",
            "stage_label": "Queued",
            "can_poll": True,
            "is_terminal": False,
        },
    }


def display_label(value: str) -> str:
    return value.replace("_", " ").title() if value else ""


def compute_display(status: str, stage: str) -> Dict[str, Any]:
    is_terminal = status in {"completed", "failed", "completed_with_errors"}
    return {
        "status_label": display_label(status),
        "stage_label": display_label(stage),
        "can_poll": not is_terminal,
        "is_terminal": is_terminal,
    }


def update_job_fields(job_id: str, **fields: Any) -> None:
    job = get_job(job_id)
    if not job:
        return

    for key, value in fields.items():
        job[key] = value

    job["updated_at"] = now_iso()
    job["display"] = compute_display(job.get("status", "queued"), job.get("stage", "queued"))
    save_job(job_id, job)


def update_job_progress(
    job_id: str,
    *,
    current_step: str,
    percent: int,
    message: str,
    steps_completed: Optional[int] = None,
) -> None:
    job = get_job(job_id)
    if not job:
        return

    progress = job.setdefault("progress", {})
    progress["current_step"] = current_step
    progress["percent"] = percent
    progress["message"] = message
    if steps_completed is not None:
        progress["steps_completed"] = steps_completed
    progress.setdefault("total_steps", 7)

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def set_job_stage(
    job_id: str,
    stage: str,
    message: str,
    steps_completed: Optional[int] = None,
) -> None:
    job = get_job(job_id)
    if not job:
        return

    job["stage"] = stage
    progress = job.setdefault("progress", {})
    progress["current_step"] = stage
    progress["percent"] = STAGE_PROGRESS.get(stage, 0)
    progress["message"] = message
    if steps_completed is not None:
        progress["steps_completed"] = steps_completed
    progress.setdefault("total_steps", 7)

    job["updated_at"] = now_iso()
    job["display"] = compute_display(job.get("status", "queued"), job.get("stage", "queued"))
    save_job(job_id, job)


def update_job_step(job_id: str, step_name: str, **fields: Any) -> None:
    job = get_job(job_id)
    if not job:
        return

    steps = job.setdefault("steps", {})
    step = steps.setdefault(step_name, build_step())

    for key, value in fields.items():
        if key == "meta":
            existing_meta = step.setdefault("meta", {})
            if isinstance(existing_meta, dict) and isinstance(value, dict):
                existing_meta.update(value)
            else:
                step["meta"] = value
        else:
            step[key] = value

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def append_job_log(job_id: str, message: str) -> None:
    job = get_job(job_id)
    if not job:
        return

    logs = job.setdefault("logs", [])
    logs.append(message)

    # Simple protection against unbounded growth.
    if len(logs) > 500:
        job["logs"] = logs[-500:]

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def set_job_output(job_id: str, output_name: str, value: Any) -> None:
    job = get_job(job_id)
    if not job:
        return

    outputs = job.setdefault("outputs", {})
    outputs[output_name] = value

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def set_job_summary(job_id: str, **fields: Any) -> None:
    job = get_job(job_id)
    if not job:
        return

    summary = job.setdefault("summary", {})
    for key, value in fields.items():
        summary[key] = value

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def add_job_error(
    job_id: str,
    *,
    code: str,
    message: str,
    step: str,
    retryable: bool = False,
) -> None:
    job = get_job(job_id)
    if not job:
        return

    errors = job.setdefault("errors", [])
    errors.append(
        {
            "code": code,
            "message": message,
            "step": step,
            "retryable": retryable,
            "timestamp": now_iso(),
        }
    )

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def append_iteration_history(job_id: str, item: Dict[str, Any]) -> None:
    job = get_job(job_id)
    if not job:
        return

    iteration = job.setdefault("iteration", {"current": 0, "max": 0, "history": []})
    history = iteration.setdefault("history", [])
    history.append(item)
    iteration["current"] = max(iteration.get("current", 0), int(item.get("iteration", 0) or 0))

    job["updated_at"] = now_iso()
    save_job(job_id, job)


def create_job(
    prompt: str,
    max_iterations: int = 3,
    user_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    job_id = str(uuid.uuid4())
    job_data = build_job_document(
        job_id=job_id,
        prompt=prompt,
        max_iterations=max_iterations,
        user_id=user_id,
        options=options,
    )
    save_job(job_id, job_data)
    return job_id


def build_runner_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged_options: Dict[str, Any] = dict(options or {})

    if FORCE_OPTIMIZER_OFF:
        merged_options["optimize_prompt"] = False
        merged_options["optimizer_enabled"] = False
        merged_options["enable_optimizer"] = False

    return merged_options


def process_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        print(f"PROCESS_JOB: job_id={job_id} not found", flush=True)
        return

    request_data = job["request"]
    prompt = request_data["prompt"]
    max_iterations = request_data["max_iterations"]
    user_id = request_data["user_id"]
    options = request_data["options"]

    start_time = time.time()
    print(f"PROCESS_JOB: started job_id={job_id}", flush=True)

    update_job_fields(
        job_id,
        status="running",
        stage="starting",
        started_at=now_iso(),
        error=None,
    )
    set_job_stage(job_id, "starting", "Workflow is starting", steps_completed=0)

    def status_callback(event: Dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "log":
            append_job_log(job_id, event.get("message", ""))

        elif event_type == "stage":
            stage = event.get("stage", "starting")
            message = event.get("message", display_label(stage))
            steps_completed = event.get("steps_completed")
            set_job_stage(job_id, stage, message, steps_completed)

        elif event_type == "step_started":
            update_job_step(
                job_id,
                event["step"],
                status="running",
                started_at=event.get("started_at", now_iso()),
                completed_at=None,
                latency_seconds=None,
                message=event.get("message"),
                error=None,
                provider=event.get("provider"),
                model=event.get("model"),
                attempts=event.get("attempts", 1),
                meta=event.get("meta", {}),
            )

        elif event_type == "step_completed":
            update_job_step(
                job_id,
                event["step"],
                status="completed",
                completed_at=event.get("completed_at", now_iso()),
                latency_seconds=event.get("latency_seconds"),
                message=event.get("message"),
                error=None,
                provider=event.get("provider"),
                model=event.get("model"),
                attempts=event.get("attempts", 1),
                meta=event.get("meta", {}),
            )

        elif event_type == "step_failed":
            update_job_step(
                job_id,
                event["step"],
                status="failed",
                completed_at=event.get("completed_at", now_iso()),
                latency_seconds=event.get("latency_seconds"),
                message=event.get("message"),
                error=event.get("error"),
                provider=event.get("provider"),
                model=event.get("model"),
                attempts=event.get("attempts", 1),
                meta=event.get("meta", {}),
            )
            add_job_error(
                job_id,
                code=event.get("code", "STEP_FAILED"),
                message=event.get("error", "Unknown step failure"),
                step=event["step"],
                retryable=event.get("retryable", False),
            )

        elif event_type == "step_skipped":
            update_job_step(
                job_id,
                event["step"],
                status="skipped",
                completed_at=event.get("completed_at", now_iso()),
                message=event.get("message"),
                meta=event.get("meta", {}),
            )

        elif event_type == "output":
            set_job_output(job_id, event["name"], event.get("value"))

        elif event_type == "summary":
            value = event.get("value", {})
            if isinstance(value, dict):
                set_job_summary(job_id, **value)

        elif event_type == "iteration":
            value = event.get("value", {})
            if isinstance(value, dict):
                append_iteration_history(job_id, value)

    try:
        prompt_length = len(prompt) if prompt else 0
        print(f"PROCESS_JOB: prompt length={prompt_length}", flush=True)
        print(f"PROCESS_JOB: max_iterations={max_iterations}", flush=True)
        print(f"PROCESS_JOB: user_id={user_id}", flush=True)
        print(f"PROCESS_JOB: options={options}", flush=True)

        result = run_autonomous_compare(
            prompt=prompt,
            max_iterations=max_iterations,
            user_id=user_id,
            options=options,
            status_callback=status_callback,
        )

        elapsed = time.time() - start_time

        if result.get("logs"):
            for line in result["logs"]:
                append_job_log(job_id, line)

        outputs = result.get("outputs", {})
        if isinstance(outputs, dict):
            for output_name, output_value in outputs.items():
                set_job_output(job_id, output_name, output_value)

        summary = result.get("summary", {})
        if isinstance(summary, dict):
            set_job_summary(job_id, **summary)

        artifacts = result.get("artifacts", [])
        html_path = None
        json_path = None
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                artifact_type = artifact.get("type")
                artifact_path = artifact.get("path")
                if artifact_type == "html":
                    html_path = artifact_path
                elif artifact_type == "json":
                    json_path = artifact_path

        job = get_job(job_id)
        if not job:
            print(f"PROCESS_JOB: job disappeared job_id={job_id}", flush=True)
            return

        if html_path is not None:
            job["artifacts"]["html_path"] = html_path
        if json_path is not None:
            job["artifacts"]["json_path"] = json_path

        runtime_seconds = result.get("runtime_seconds")
        if runtime_seconds is None:
            runtime_seconds = round(elapsed, 2)
        job["timing"]["total_runtime_seconds"] = runtime_seconds

        runner_status = result.get("status", "completed")
        top_level_error = result.get("error")

        if runner_status == "completed":
            job["status"] = "completed" if not job.get("errors") else "completed_with_errors"
            job["stage"] = "completed"
            job["completed_at"] = now_iso()
            job["error"] = None
            job["progress"] = {
                "current_step": "completed",
                "percent": 100,
                "message": "Workflow completed successfully",
                "steps_completed": 7,
                "total_steps": 7,
            }
        else:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["completed_at"] = now_iso()
            job["error"] = {
                "code": "WORKFLOW_FAILED",
                "message": top_level_error or "Workflow failed",
                "step": job.get("stage", "failed"),
                "retryable": False,
            }
            job["progress"] = {
                "current_step": "failed",
                "percent": 100,
                "message": top_level_error or "Workflow failed",
                "steps_completed": job.get("progress", {}).get("steps_completed", 0),
                "total_steps": 7,
            }

        job["updated_at"] = now_iso()
        job["display"] = compute_display(job.get("status", "queued"), job.get("stage", "queued"))
        save_job(job_id, job)

        print(
            f"PROCESS_JOB: completed job_id={job_id} "
            f"status={job['status']} in {elapsed:.2f}s",
            flush=True,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"PROCESS_JOB ERROR: job_id={job_id} failed after {elapsed:.2f}s", flush=True)
        print(f"PROCESS_JOB ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

        job = get_job(job_id)
        if job:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["completed_at"] = now_iso()
            job["error"] = {
                "code": "PROCESS_JOB_EXCEPTION",
                "message": str(e),
                "step": "process_job",
                "retryable": False,
            }
            job.setdefault("errors", []).append(
                {
                    "code": "PROCESS_JOB_EXCEPTION",
                    "message": str(e),
                    "step": "process_job",
                    "retryable": False,
                    "timestamp": now_iso(),
                }
            )
            job["timing"]["total_runtime_seconds"] = round(elapsed, 2)
            job["progress"] = {
                "current_step": "failed",
                "percent": 100,
                "message": str(e),
                "steps_completed": job.get("progress", {}).get("steps_completed", 0),
                "total_steps": 7,
            }
            job["updated_at"] = now_iso()
            job["display"] = compute_display(job.get("status", "queued"), job.get("stage", "queued"))
            save_job(job_id, job)


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def serve_index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status/{job_id}")
def get_status(job_id: str) -> Dict[str, Any]:
    print(f"API: /status called for job_id={job_id} pid={os.getpid()}", flush=True)

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/run")
def run_workflow(request: RunRequest):
    print(f"API: /run called pid={os.getpid()}", flush=True)

    try:
        prompt_length = len(request.prompt) if request.prompt else 0
        print(f"API: /run prompt length={prompt_length}", flush=True)
        print(f"API: /run max_iterations={request.max_iterations}", flush=True)
        print(f"API: /run user_id={request.user_id}", flush=True)

        options = build_runner_options(request.options)
        print(f"API: /run options={options}", flush=True)

        job_id = create_job(
            prompt=request.prompt,
            max_iterations=request.max_iterations,
            user_id=request.user_id,
            options=options,
        )

        print(f"API: /run created job_id={job_id} pid={os.getpid()}", flush=True)

        thread = threading.Thread(
            target=process_job,
            args=(job_id,),
            daemon=True,
        )
        thread.start()

        print(f"API: /run background thread started for job_id={job_id}", flush=True)

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Workflow started successfully",
        }

    except Exception as e:
        print("API ERROR: /run failed before job start", flush=True)
        print(f"API ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "endpoint": "/run",
            },
        )


def build_file_context(file_payloads: List[Dict[str, str]]) -> str:
    if not file_payloads:
        return ""

    sections = []
    for item in file_payloads:
        sections.append(
            f"""Filename: {item["filename"]}
Content:
{item["content"]}"""
        )

    return "\n\n--- ATTACHED FILES CONTEXT ---\n\n" + "\n\n====\n\n".join(sections)


@app.post("/run-with-files")
async def run_workflow_with_files(
    prompt: str = Form(...),
    max_iterations: int = Form(3),
    user_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    start_time = time.time()
    print("API: /run-with-files called")

    try:
        print(f"API: /run-with-files prompt length={len(prompt) if prompt else 0}")
        print(f"API: /run-with-files file count={len(files)}")
        print(f"API: /run-with-files max_iterations={max_iterations}")
        print(f"API: /run-with-files user_id={user_id}")

        file_payloads: List[Dict[str, str]] = []

        for uploaded_file in files:
            print(f"API: reading uploaded file={uploaded_file.filename}")
            raw = await uploaded_file.read()

            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = raw.decode("latin-1")
                except UnicodeDecodeError:
                    content = (
                        "[Binary or unsupported text encoding. File uploaded successfully, "
                        "but content could not be decoded as text.]"
                    )

            file_payloads.append(
                {
                    "filename": uploaded_file.filename or "unnamed_file",
                    "content": content[:20000],
                }
            )

        combined_prompt = prompt + build_file_context(file_payloads)

        options = build_runner_options(
            {
                "attached_filenames": [f["filename"] for f in file_payloads]
            }
        )
        print(f"API: /run-with-files options={options}")
        print("API: /run-with-files starting runner")

        result = run_autonomous_compare(
            prompt=combined_prompt,
            max_iterations=max_iterations,
            user_id=user_id,
            options=options,
            status_callback=None,
        )

        result["attached_files"] = [f["filename"] for f in file_payloads]

        elapsed = time.time() - start_time
        print(
            f"API: /run-with-files completed with status={result.get('status')} "
            f"in {elapsed:.2f}s"
        )
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"API ERROR: /run-with-files failed after {elapsed:.2f}s")
        print(f"API ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "endpoint": "/run-with-files",
                "elapsed_seconds": round(elapsed, 2),
            },
        )


@app.post("/run-test")
def run_test(request: RunRequest) -> Dict[str, Any]:
    print("API: /run-test called")
    return {
        "status": "ok",
        "message": "run-test endpoint reached successfully",
        "prompt_length": len(request.prompt) if request.prompt else 0,
        "max_iterations": request.max_iterations,
        "user_id": request.user_id,
    }
