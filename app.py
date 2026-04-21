from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autonomous_compare_runner import run_autonomous_compare

app = FastAPI(title="Autonomous Workflow API")

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "index.html"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


class RunRequest(BaseModel):
    prompt: str
    max_iterations: int = 3
    user_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run_workflow(request: RunRequest) -> Dict[str, Any]:
    return run_autonomous_compare(
        prompt=request.prompt,
        max_iterations=request.max_iterations,
        user_id=request.user_id,
        options=request.options,
    )
