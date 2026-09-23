from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import shutil
import tempfile
import time
import uuid
import os

from ..nlp.script_analyzer import ScriptAnalyzer
from ..vision.storyboard_generator import StoryboardGenerator

app = FastAPI(title="Script-to-Storyboard API")

# Initialize components
script_analyzer = ScriptAnalyzer()
storyboard_generator = StoryboardGenerator(backend=os.getenv("STORYBOARD_BACKEND", "auto"))

# One worker: the image model can only draw one storyboard at a time, so jobs queue up.
# ponytail: in-memory job table, lost on restart; move to a real queue if many people use this at once.
worker = ThreadPoolExecutor(max_workers=1)
jobs = {}
JOB_TTL_SECONDS = 3600

UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Script2Storyboard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 34rem; margin: 4rem auto; padding: 0 1rem; line-height: 1.5; }
  button { padding: .5rem 1.25rem; font-size: 1rem; cursor: pointer; }
  progress { width: 100%; height: 1rem; }
  [hidden] { display: none; }
</style></head>
<body>
  <h1>Script2Storyboard</h1>
  <p>Upload a screenplay (<code>.txt</code>, with <code>INT.</code> / <code>EXT.</code> scene headings).
     You'll get the storyboard as a PDF, one frame per shot.</p>
  <form id="form">
    <p><label for="file">Script file</label><br>
       <input id="file" name="file" type="file" accept=".txt,text/plain" required></p>
    <button id="go" type="submit">Generate storyboard</button>
  </form>
  <progress id="bar" hidden></progress>
  <p id="status" role="status" aria-live="polite"></p>
<script>
const form = document.getElementById("form"), go = document.getElementById("go");
const bar = document.getElementById("bar"), status = document.getElementById("status");

function finish(message, link) {
  go.disabled = false;
  bar.hidden = true;
  status.textContent = message;
  if (link) {
    const a = document.createElement("a");
    a.href = link;
    a.textContent = "Download storyboard PDF";
    status.append(" ", a);
    window.location.href = link;
  }
}

async function poll(id) {
  const job = await (await fetch("/jobs/" + id)).json();
  if (job.status === "done") return finish("Done.", "/jobs/" + id + "/pdf");
  if (job.status === "failed") return finish("Failed: " + job.error);
  bar.max = job.total;
  bar.value = job.done;
  status.textContent = job.status === "queued"
    ? "Waiting for the previous storyboard to finish..."
    : "Drawing frame " + (job.done + 1) + " of " + job.total + "...";
  setTimeout(() => poll(id), 2000);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  go.disabled = true;
  bar.hidden = false;
  bar.removeAttribute("value");
  status.textContent = "Reading script...";
  try {
    const response = await fetch("/jobs", { method: "POST", body: new FormData(form) });
    const job = await response.json();
    if (!response.ok) return finish(job.detail || "Upload failed.");
    poll(job.id);
  } catch (error) {
    finish("Could not reach the server.");
  }
});
</script>
</body>
</html>"""


def read_scenes(file: UploadFile):
    """Decode and parse an uploaded script, rejecting bad input with a 400."""
    try:
        script_text = file.file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Script must be a UTF-8 text file.")
    scenes = script_analyzer.process_script(script_text)
    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes found in the script.")
    return scenes


def build_pdf(scenes, out_dir: str, on_progress=None) -> str:
    image_paths = storyboard_generator.generate_storyboard(
        [asdict(scene) for scene in scenes], output_dir=out_dir, on_progress=on_progress
    )
    pdf_path = os.path.join(out_dir, "storyboard.pdf")
    storyboard_generator.create_storyboard_pdf(image_paths, pdf_path)
    return pdf_path


def purge_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id, job in list(jobs.items()):
        if job.get("finished_at", time.time()) < cutoff:
            shutil.rmtree(job["dir"], ignore_errors=True)
            jobs.pop(job_id, None)


def public(job: dict) -> dict:
    return {k: job[k] for k in ("status", "done", "total", "error")}


@app.get("/", response_class=HTMLResponse)
def upload_page():
    """Browser UI: upload a script, watch progress, download the storyboard PDF."""
    return UPLOAD_PAGE


@app.post("/analyze-script")
def analyze_script(file: UploadFile = File(...)):
    """Analyze a script file and return scene analysis."""
    return {"scenes": [asdict(scene) for scene in read_scenes(file)]}


@app.post("/jobs")
def start_job(file: UploadFile = File(...)):
    """Queue a storyboard; poll GET /jobs/{id} for progress, then GET /jobs/{id}/pdf."""
    purge_old_jobs()
    scenes = read_scenes(file)
    total = sum(len(StoryboardGenerator.frames_for(asdict(scene))) for scene in scenes)
    job_id = uuid.uuid4().hex
    job = jobs[job_id] = {"status": "queued", "done": 0, "total": total, "error": None,
                          "dir": tempfile.mkdtemp(), "pdf": None}

    def run():
        job["status"] = "running"
        try:
            job["pdf"] = build_pdf(scenes, job["dir"], lambda done, _total: job.update(done=done))
            job["status"] = "done"
        except Exception as e:
            job.update(status="failed", error=str(e))
        job["finished_at"] = time.time()

    worker.submit(run)
    return {"id": job_id, **public(job)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return public(job)


@app.get("/jobs/{job_id}/pdf")
def job_pdf(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Storyboard not ready.")
    return FileResponse(job["pdf"], media_type="application/pdf", filename="storyboard.pdf")


@app.post("/generate-storyboard")
def generate_storyboard(file: UploadFile = File(...)):
    """Generate a storyboard and return the PDF in one request (blocks until done)."""
    scenes = read_scenes(file)
    temp_dir = tempfile.mkdtemp()
    try:
        # Runs on the same single worker as /jobs, so the two never draw at once.
        pdf_path = worker.submit(build_pdf, scenes, temp_dir).result()
        # The temp dir is removed after the response is sent, not before.
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="storyboard.pdf",
            background=BackgroundTask(shutil.rmtree, temp_dir, ignore_errors=True)
        )
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
