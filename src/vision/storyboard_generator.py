import base64
import io
import os
import re
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import requests
except ImportError:  # pragma: no cover - exercised when requests is unavailable
    requests = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None

try:
    from diffusers import StableDiffusionPipeline
except ImportError:  # pragma: no cover - exercised when diffusers is unavailable
    StableDiffusionPipeline = None


def _font(size: int):
    # Pillow >= 10.1 ships a scalable default font, so no font files are needed.
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, max_width: int, max_lines: int) -> List[str]:
    """Word-wrap text to a pixel width, ending with an ellipsis if it doesn't fit."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if font.getlength(candidate) <= max_width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and font.getlength(lines[-1] + "...") > max_width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0] if " " in lines[-1] else lines[-1][:-1]
        lines[-1] += "..."
    return lines


class StoryboardGenerator:
    def __init__(self, backend: str = "auto"):
        self._load_env_file()

        self.text_to_image = None
        self._stable_diffusion_device = None

        # Resolve once, so a missing model is reported (and retried) once, not per frame.
        self.backend = (backend or "auto").strip().lower()
        self.backend = self._resolve_backend()
        print(f"Image backend: {self.backend}")

    def _load_env_file(self) -> None:
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]
        for path in candidate_paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line or line.startswith("#") or "=" not in line:
                                continue
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            except OSError:
                continue

    def _resolve_backend(self) -> str:
        if self.backend in {"fallback", "none"}:
            return "fallback"
        # "Nano Banana" is Google's name for Gemini 2.5 Flash Image, so it's the same backend.
        if self.backend in {"gemini", "google", "google-gemini", "nano-banana", "nano_banana", "banana"}:
            if not self._gemini_available():
                print("Gemini backend needs GEMINI_API_KEY; using the local renderer")
                return "fallback"
            return "gemini"
        if self.backend in {"stable-diffusion", "sd", "diffusion"}:
            return "stable-diffusion" if self._ensure_stable_diffusion() else "fallback"
        if self._ensure_stable_diffusion():
            return "stable-diffusion"
        if self._gemini_available():
            return "gemini"
        return "fallback"

    def _gemini_available(self) -> bool:
        return bool(self._get_gemini_api_key()) and requests is not None

    def _get_gemini_api_key(self) -> str:
        for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
            value = os.getenv(key_name)
            if value:
                return value
        return ""

    def _ensure_stable_diffusion(self) -> bool:
        if self.text_to_image is not None:
            return True
        if StableDiffusionPipeline is None or torch is None:
            return False
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.text_to_image = StableDiffusionPipeline.from_pretrained(
                os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
                torch_dtype=dtype,
            )
            self.text_to_image = self.text_to_image.to(device)
            self.text_to_image.enable_attention_slicing()  # lower peak memory, small speed cost
            self._stable_diffusion_device = device
            if device == "cpu":
                print("Stable Diffusion is running on CPU: expect a few minutes per frame. "
                      "See README for installing the CUDA build of PyTorch.")
            return True
        except Exception as e:
            print(f"Stable Diffusion unavailable ({type(e).__name__}: {e}); using the local renderer")
            self.text_to_image = None
            self._stable_diffusion_device = None
            return False

    def _build_stable_diffusion_prompt(self, scene_description: str) -> str:
        # CLIP reads only ~77 tokens, so keep the style tags short and the scene first.
        clean_prompt = scene_description.strip() or "cinematic scene"
        return f"storyboard frame, pencil sketch, {clean_prompt}, cinematic composition"

    @staticmethod
    def scene_prompt(scene: dict) -> str:
        """Describe what the camera sees: location, action and mood, not the dialogue."""
        heading = scene.get("heading", "")
        location = re.sub(r"^\s*(?:INT\.?/EXT|EXT\.?/INT|I/E|INT|EXT)[.\s-]*", "", heading, flags=re.IGNORECASE)
        parts = [location.strip(" -.").lower(), scene.get("action_text", "")]
        emotions = [e for e in scene.get("emotions", []) if e != "neutral"]
        if emotions:
            parts.append(f"{emotions[0]} mood")
        prompt = ", ".join(p.strip(" .") for p in parts if p.strip(" ."))
        return prompt or scene.get("content", "")

    def _image_from_bytes(self, image_bytes: bytes, size: Tuple[int, int]) -> np.ndarray:
        if not image_bytes:
            return None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                img = img.resize(size)
                return np.array(img)
        except Exception:
            return None

    def _extract_gemini_image(self, data: dict, size: Tuple[int, int]) -> np.ndarray:
        if not isinstance(data, dict):
            return None

        for candidate in data.get("candidates", []):
            content = candidate.get("content") or {}
            for part in content.get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data") or {}
                if isinstance(inline_data, dict):
                    image_data = inline_data.get("data")
                    if isinstance(image_data, str):
                        if image_data.startswith("data:image"):
                            image_data = image_data.split(",", 1)[1]
                        try:
                            return self._image_from_bytes(base64.b64decode(image_data), size)
                        except Exception:
                            continue
        return None

    def _generate_with_gemini(self, scene_description: str, size: Tuple[int, int]) -> np.ndarray:
        api_key = self._get_gemini_api_key()
        if not api_key:
            return None

        api_url = os.getenv("GEMINI_API_URL") or "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Black-and-white pencil storyboard frame, cinematic composition, no text: {scene_description}"}],
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        try:
            response = requests.post(api_url, headers={"x-goog-api-key": api_key}, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Gemini request failed: {e}")
            return None

        return self._extract_gemini_image(data, size)

    def _build_fallback_scene(self, scene_description: str, size: Tuple[int, int]) -> np.ndarray:
        """Empty frame used when no image model ran; the caption below still tells the scene."""
        width, height = size
        frame = Image.new("RGB", (width, height), (238, 237, 232))
        draw = ImageDraw.Draw(frame)
        draw.line([(0, 0), (width, height)], fill=(222, 221, 215), width=2)
        draw.line([(0, height), (width, 0)], fill=(222, 221, 215), width=2)
        location = scene_description.split(",")[0].strip().upper()
        font, small = _font(26), _font(15)
        lines = _wrap(location, font, width - 80, 2)
        y = height // 2 - 20 * len(lines)
        for line in lines:
            draw.text(((width - font.getlength(line)) / 2, y), line, font=font, fill=(90, 90, 90))
            y += 36
        note = "no image model - use --backend stable-diffusion or gemini"
        draw.text(((width - small.getlength(note)) / 2, height - 40), note, font=small, fill=(150, 150, 150))
        return np.array(frame)

    def generate_storyboard_frame(self, scene_description: str, 
                                size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Generate a storyboard frame from scene description."""
        backend = self._resolve_backend()
        if backend == "gemini":
            image = self._generate_with_gemini(scene_description, size)
            if image is not None:
                return image
            print("No image from Gemini; falling back to local renderer")

        if backend == "stable-diffusion":
            if self._ensure_stable_diffusion():
                prompt = self._build_stable_diffusion_prompt(scene_description)
                result = self.text_to_image(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    negative_prompt="blurry, low quality, distorted, text, watermark, duplicate",
                    # Same seed for every frame keeps the storyboard's look consistent.
                    generator=torch.Generator(self._stable_diffusion_device).manual_seed(int(os.getenv("SD_SEED", "42"))),
                    height=size[1],
                    width=size[0],
                )
                image = result.images[0]
                return np.array(image)

        return self._build_fallback_scene(scene_description, size)

    CAPTION_HEIGHT = 170

    def add_storyboard_elements(self, image: np.ndarray, scene_info: dict) -> np.ndarray:
        """Build a storyboard panel: the frame, with shot number, slugline, action and dialogue below it."""
        frame = Image.fromarray(image).convert("RGB")
        width, pad = frame.width, 16
        panel = Image.new("RGB", (width, frame.height + self.CAPTION_HEIGHT), "white")
        panel.paste(frame, (0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([0, 0, width - 1, frame.height - 1], outline=(20, 20, 20), width=2)

        heading = scene_info.get("heading") or ""
        action = scene_info.get("action_text") or ("" if heading else scene_info.get("content", ""))
        dialogue = scene_info.get("dialogue") or []
        title_font, body_font = _font(20), _font(16)
        text_width = width - 2 * pad

        y = frame.height + 12
        shot = str(scene_info.get("id", ""))
        draw.text((pad, y), shot, font=title_font, fill=(200, 60, 40))
        shot_w = int(title_font.getlength(shot)) + 12
        for line in _wrap(heading.upper(), title_font, text_width - shot_w, 1):
            draw.text((pad + shot_w, y), line, font=title_font, fill=(20, 20, 20))
        y += 32
        for line in _wrap(action, body_font, text_width, 3 if dialogue else 5):
            draw.text((pad, y), line, font=body_font, fill=(40, 40, 40))
            y += 22
        if dialogue:
            y += 6
            quoted = [f'{who}: "{said}"' for who, _, said in (d.partition(": ") for d in dialogue)]
            for line in _wrap("   ".join(quoted), body_font, text_width, 2):
                draw.text((pad, y), line, font=body_font, fill=(110, 110, 110))
                y += 22
        return np.array(panel)

    def generate_storyboard(self, scenes: List[dict], 
                          output_dir: str = "output") -> List[str]:
        """Generate complete storyboard from list of scenes."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for index, scene in enumerate(scenes, 1):
            image = self.generate_storyboard_frame(self.scene_prompt(scene))
            image = self.add_storyboard_elements(image, scene)
            output_path = os.path.join(output_dir, f"scene_{index:03d}.png")
            Image.fromarray(image).save(output_path)
            output_paths.append(output_path)
            
        return output_paths

    def create_storyboard_pdf(self, image_paths: List[str], output_path: str,
                              title: str = "Storyboard") -> str:
        """Lay panels out on landscape sheets, 3 x 2 per page, and save them as a PDF."""
        if not image_paths:
            raise ValueError("No images were provided for PDF generation.")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        page_w, page_h, margin, gap, header = 2480, 1754, 90, 50, 90  # A4 landscape at 212 dpi
        cols, rows = 3, 2
        cell_w = (page_w - 2 * margin - (cols - 1) * gap) // cols
        cell_h = (page_h - 2 * margin - header - (rows - 1) * gap) // rows
        per_page = cols * rows
        page_count = (len(image_paths) + per_page - 1) // per_page

        pages = []
        for p in range(page_count):
            page = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(page)
            draw.text((margin, margin), title.upper(), font=_font(40), fill=(20, 20, 20))
            label = f"{p + 1} / {page_count}"
            draw.text((page_w - margin - _font(28).getlength(label), margin + 8), label, font=_font(28), fill=(120, 120, 120))
            for k, img_path in enumerate(image_paths[p * per_page:(p + 1) * per_page]):
                with Image.open(img_path) as img:
                    panel = img.convert("RGB")
                scale = min(cell_w / panel.width, cell_h / panel.height)
                panel = panel.resize((int(panel.width * scale), int(panel.height * scale)), Image.LANCZOS)
                x = margin + (k % cols) * (cell_w + gap) + (cell_w - panel.width) // 2
                y = margin + header + (k // cols) * (cell_h + gap)
                page.paste(panel, (x, y))
                draw.rectangle([x - 1, y - 1, x + panel.width, y + panel.height], outline=(200, 200, 200), width=2)
            pages.append(page)

        pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=212.0)
        return output_path
