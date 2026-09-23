# Limitations and Fixes

## Addressed

1. **CLIP token length**: Stable Diffusion's text encoder reads only 77 tokens. The script is now parsed as a screenplay (headings, character cues, dialogue, parentheticals), and the image prompt uses only the location, the action lines and the mood. Dialogue, which the camera can't see, is left out, so a typical scene fits.

2. **GPU dependency**: Stable Diffusion runs on CPU when CUDA isn't available (slower), with attention slicing to lower memory. `--backend fallback` renders simple frames with no GPU or model download, and `--backend gemini` uses a hosted model.

3. **Model compatibility**: `requirements.txt` lists only the packages the code imports. Unused TensorFlow, Keras and TensorBoard were removed. The Stable Diffusion checkpoint can be swapped with `SD_MODEL_ID`.

4. **Error handling**: The API rejects non-UTF-8 files and scripts with no scenes with a 400 response. The emotion classifier truncates long scenes instead of crashing. Failed Gemini or Nano Banana requests fall back to the local renderer.

5. **User interface**: `uvicorn src.api.main:app` serves an upload page at `/`. Pick a script and get the storyboard PDF back, with no command line needed.

## Remaining

1. **Very long scripts**: Frames are generated one at a time, and the PDF is built from all frames at the end. A feature-length script works but is slow on CPU. Batching several prompts per Stable Diffusion call would help on a GPU.

2. **Character consistency**: A shared seed keeps the style consistent, but the same character can look different from frame to frame. Fixing that needs a reference-image approach (for example IP-Adapter).

3. **Long single scenes**: A scene with a very long action block can still exceed 77 tokens, and the end of the prompt is cut off. Splitting such scenes into several shots, one frame per paragraph, would fix this.
