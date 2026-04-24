from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import time
import traceback
import uuid
import threading
import os

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

# Temporary in-memory job store
# Later this can be replaced with Redis
jobs: Dict[str, Dict[str, Any]] = {}


class RunRequest(BaseModel):
    prompt: str
    max_iterations: int = 3
    user_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(
    prompt: str,
    max_iterations: int = 3,
    user_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "request": {
            "prompt": prompt,
            "max_iterations": max_iterations,
            "user_id": user_id,
            "options": options or {},
        },
        "results": {
            "openai": {
                "status": "pending",
                "output": None,
                "latency_seconds": None,
                "error": None,
            },
            "claude": {
                "status": "pending",
                "output": None,
                "latency_seconds": None,
                "error": None,
            },
            "judge": {
                "status": "pending",
                "output": None,
                "latency_seconds": None,
                "error": None,
            },
        },
        "runner_result": None,
        "final_output": None,
        "artifacts": {
            "html_path": None,
            "json_path": None,
        },
        "error": None,
    }

    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    for key, value in fields.items():
        job[key] = value

    job["updated_at"] = now_iso()


def build_runner_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged_options: Dict[str, Any] = dict(options or {})

    if FORCE_OPTIMIZER_OFF:
        merged_options["optimize_prompt"] = False
        merged_options["optimizer_enabled"] = False
        merged_options["enable_optimizer"] = False

    return merged_options


def process_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        print(f"PROCESS_JOB: job_id={job_id} not found")
        return

    request_data = job["request"]
    prompt = request_data["prompt"]
    max_iterations = request_data["max_iterations"]
    user_id = request_data["user_id"]
    options = request_data["options"]

    start_time = time.time()
    print(f"PROCESS_JOB: started job_id={job_id}")

    try:
        update_job(job_id, status="running", stage="starting")

        prompt_length = len(prompt) if prompt else 0
        print(f"PROCESS_JOB: prompt length={prompt_length}")
        print(f"PROCESS_JOB: max_iterations={max_iterations}")
        print(f"PROCESS_JOB: user_id={user_id}")
        print(f"PROCESS_JOB: options={options}")

        update_job(job_id, stage="running_runner")

        result = run_autonomous_compare(
            prompt=prompt,
            max_iterations=max_iterations,
            user_id=user_id,
            options=options,
        )

        elapsed = time.time() - start_time

        jobs[job_id]["runner_result"] = result

        jobs[job_id]["final_output"] = (
            result.get("final_output")
            or result.get("full_output")
            or result.get("output")
            or result.get("response")
            or result.get("best_response")
        )

        artifacts = result.get("artifacts", [])

        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue

                artifact_type = artifact.get("type")
                artifact_path = artifact.get("path")

                if artifact_type == "html":
                    jobs[job_id]["artifacts"]["html_path"] = artifact_path
                elif artifact_type == "json":
                    jobs[job_id]["artifacts"]["json_path"] = artifact_path

        print(f"PROCESS_JOB: BASE_DIR={BASE_DIR}", flush=True)
        print(f"PROCESS_JOB: OUTPUTS_DIR={OUTPUTS_DIR}", flush=True)
        print(
            f"PROCESS_JOB: extracted html_path={jobs[job_id]['artifacts']['html_path']}",
            flush=True,
        )
        print(
            f"PROCESS_JOB: extracted json_path={jobs[job_id]['artifacts']['json_path']}",
            flush=True,
        )

        jobs[job_id]["status"] = result.get("status", "completed")
        jobs[job_id]["stage"] = "done"
        jobs[job_id]["updated_at"] = now_iso()

        print(
            f"PROCESS_JOB: completed job_id={job_id} "
            f"status={jobs[job_id]['status']} in {elapsed:.2f}s"
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"PROCESS_JOB ERROR: job_id={job_id} failed after {elapsed:.2f}s")
        print(f"PROCESS_JOB ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

        jobs[job_id]["status"] = "failed"
        jobs[job_id]["stage"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["updated_at"] = now_iso()


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def serve_index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status/{job_id}")
def get_status(job_id: str) -> Dict[str, Any]:
    print(f"API: /status called for job_id={job_id} pid={os.getpid()}", flush=True)
    print(f"API: /status jobs_count={len(jobs)}", flush=True)
    print(f"API: /status current_job_ids={list(jobs.keys())}", flush=True)

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/run")
def run_workflow(request: RunRequest):
    print(f"API: /run called pid={os.getpid()}", flush=True)
    print(f"API: /run jobs_before={len(jobs)}", flush=True)

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
        print(f"API: /run jobs_after={len(jobs)}", flush=True)
        print(f"API: /run current_job_ids={list(jobs.keys())}", flush=True)

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
