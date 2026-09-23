# Limitations and Fixes

## Addressed

1. **CLIP token length**: Stable Diffusion's text encoder reads only 77 tokens. The script is parsed as a screenplay, and each action paragraph becomes its own shot, so a prompt only has to describe one moment: location, camera framing, action and mood. Dialogue goes in the caption, not the prompt.

2. **GPU dependency**: On Windows the default PyTorch wheel is CPU-only. With the CUDA build (see README) a 4 GB laptop GPU draws a frame in seconds. On CPU it still works, but slowly, and the program says so. `--backend gemini` uses a hosted model, and `--backend fallback` produces the layout with placeholder frames.

3. **Model compatibility**: `requirements.txt` lists only the packages the code imports, with minimum versions for the Starlette and AnyIO security fixes. The Stable Diffusion checkpoint can be swapped with `SD_MODEL_ID`.

4. **Error handling**: The API rejects non-UTF-8 files and scripts with no scenes with a 400 response. The emotion classifier truncates long text instead of crashing. If Stable Diffusion, IP-Adapter or Gemini fails, the program prints the reason before falling back.

5. **User interface and long scripts**: `uvicorn src.api.main:app` serves an upload page that queues the job, shows frame-by-frame progress and downloads the PDF when it's done, so long scripts don't hit browser timeouts.

6. **Executable**: `python build.py` makes a folder build with the spaCy model bundled. The old one-file build unpacked PyTorch on every launch and loaded the spaCy model twice at startup.

## Remaining

1. **Characters don't look the same from frame to frame**: All frames share a pencil-sketch style, but Sarah in scene 1 isn't recognisably Sarah in scene 3. Using the scene's first frame as an IP-Adapter reference (`SD_REFERENCE_STRENGTH`) was tested on the sample script. Any strength strong enough to keep faces similar also copied the first frame's composition, so close-ups came out as wide shots, and it's off by default. A real fix needs one reference portrait per character (for example IP-Adapter FaceID), or a newer model that takes a character reference directly.

2. **Camera framing is a heuristic when the script gives none**: Explicit directions (`CLOSE ON`, `WIDE SHOT`, `ANGLE ON`, `INSERT`, `POV`) are respected. Otherwise the first shot of a scene is wide, shots with dialogue are medium or two-shots, and short beats are close-ups.

3. **Jobs are kept in memory**: API jobs run one at a time and are lost if the server restarts. That's fine for one user, but many users need a real job queue.
