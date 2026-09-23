from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from dataclasses import asdict
import shutil
import tempfile
import os

from ..nlp.script_analyzer import ScriptAnalyzer
from ..vision.storyboard_generator import StoryboardGenerator

app = FastAPI(title="Script-to-Storyboard API")

# Initialize components
script_analyzer = ScriptAnalyzer()
storyboard_generator = StoryboardGenerator()

UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Script2Storyboard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 32rem; margin: 4rem auto; padding: 0 1rem; line-height: 1.5; }
  button { padding: .5rem 1.25rem; font-size: 1rem; cursor: pointer; }
</style></head>
<body>
  <h1>Script2Storyboard</h1>
  <p>Upload a screenplay (<code>.txt</code>, with <code>INT.</code> / <code>EXT.</code> scene headings).
     You'll get the storyboard as a PDF. With Stable Diffusion on CPU this can take several minutes.</p>
  <form action="/generate-storyboard" method="post" enctype="multipart/form-data"
        onsubmit="this.querySelector('button').textContent = 'Generating...'">
    <p><label for="file">Script file</label><br>
       <input id="file" name="file" type="file" accept=".txt,text/plain" required></p>
    <button type="submit">Generate storyboard</button>
  </form>
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


@app.get("/", response_class=HTMLResponse)
def upload_page():
    """Minimal browser UI: upload a script, download the storyboard PDF."""
    return UPLOAD_PAGE


@app.post("/analyze-script")
def analyze_script(file: UploadFile = File(...)):
    """Analyze a script file and return scene analysis."""
    return {"scenes": [asdict(scene) for scene in read_scenes(file)]}

@app.post("/generate-storyboard")
def generate_storyboard(file: UploadFile = File(...)):
    """Generate storyboard from script file."""
    # Plain def: FastAPI runs it in a worker thread, so image generation doesn't block the server.
    # The temp dir is removed after the response is sent, not before.
    scenes = read_scenes(file)
    temp_dir = tempfile.mkdtemp()
    try:
        image_paths = storyboard_generator.generate_storyboard(
            [asdict(scene) for scene in scenes],
            output_dir=temp_dir
        )
        pdf_path = os.path.join(temp_dir, "storyboard.pdf")
        storyboard_generator.create_storyboard_pdf(image_paths, pdf_path)
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