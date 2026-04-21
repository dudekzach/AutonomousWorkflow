from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
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
) -> Dict[str, Any]:
    file_payloads: List[Dict[str, str]] = []

    for uploaded_file in files:
        raw = await uploaded_file.read()

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("latin-1")
            except UnicodeDecodeError:
                content = "[Binary or unsupported text encoding. File uploaded successfully, but content could not be decoded as text.]"

        file_payloads.append(
            {
                "filename": uploaded_file.filename or "unnamed_file",
                "content": content[:20000],  # keep first 20k chars per file for now
            }
        )

    combined_prompt = prompt + build_file_context(file_payloads)

    result = run_autonomous_compare(
        prompt=combined_prompt,
        max_iterations=max_iterations,
        user_id=user_id,
        options={
            "attached_filenames": [f["filename"] for f in file_payloads]
        },
    )

    result["attached_files"] = [f["filename"] for f in file_payloads]
    return result
