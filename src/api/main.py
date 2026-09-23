from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
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

@app.post("/analyze-script")
def analyze_script(file: UploadFile = File(...)):
    """Analyze a script file and return scene analysis."""
    try:
        script_text = file.file.read().decode()
        
        # Process script
        scenes = script_analyzer.process_script(script_text)
        
        return {"scenes": [asdict(scene) for scene in scenes]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-storyboard")
def generate_storyboard(file: UploadFile = File(...)):
    """Generate storyboard from script file."""
    # Plain def: FastAPI runs it in a worker thread, so image generation doesn't block the server.
    # The temp dir is removed after the response is sent, not before.
    temp_dir = tempfile.mkdtemp()
    try:
        script_text = file.file.read().decode()
        scenes = script_analyzer.process_script(script_text)
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