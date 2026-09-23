# Script-to-Storyboard Generation System

An AI-powered system that converts film scripts into visual storyboards using NLP, computer vision, and deep learning.

## Features

- Script parsing and scene segmentation
- Emotion and action analysis
- Scene classification for dialogue and action moments
- Visual storyboard generation with generated frames
- Example script and output assets included in the repository

## Example Output

The repository now includes a sample input script and a generated storyboard example.

![Sample storyboard frame](examples/scene_001.png)

- Sample input script: [examples/sample_script.txt](examples/sample_script.txt)
- Generated storyboard PDF: [examples/storyboard-example.pdf](examples/storyboard-example.pdf)
- Additional storyboard frames: [examples/scene_001.png](examples/scene_001.png) to [examples/scene_006.png](examples/scene_006.png)

## Project Structure

```text
script2storyboard/
├── src/
│   ├── nlp/              # NLP processing modules
│   ├── vision/           # Computer vision and image generation
│   └── api/              # API endpoints and services
├── examples/            # Example scripts and generated storyboard assets
├── tests/               # Unit tests
└── output/              # Generated storyboard outputs
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model if needed:
```bash
python -m spacy download en_core_web_lg
```

## Usage

Run the generator with any script file:

```bash
python src/main.py --script path/to/script.txt
```

The generated storyboard PDF will be written to the output folder.

Options:

- `--backend auto|stable-diffusion|gemini|nano-banana|fallback`: image backend. `auto` picks the first one available. `fallback` needs no GPU or API key.
- `--output DIR`: output folder (default `output`).
- `--no-open`: don't open the PDF when done.

Environment variables (can go in a `.env` file): `GEMINI_API_KEY` for the Gemini backend, `SD_MODEL_ID` to use a different Stable Diffusion checkpoint, `SD_SEED` to change the seed shared by all frames (default 42, which keeps the frames visually consistent).

The script is parsed as a screenplay: scene headings (`INT.` / `EXT.`), character cues and dialogue are separated, so the image prompt describes only what the camera sees (location, action, mood). This keeps it short enough for Stable Diffusion's 77-token CLIP limit.

### API

```bash
uvicorn src.api.main:app
```

Open http://127.0.0.1:8000 for a simple upload page: pick a script and get the storyboard PDF back.

`POST /analyze-script` returns the scene analysis as JSON. `POST /generate-storyboard` returns the PDF. Both take the script as a multipart `file` upload.

### Tests

```bash
python -m unittest tests.test_storyboard_generator
```

## License

MIT License