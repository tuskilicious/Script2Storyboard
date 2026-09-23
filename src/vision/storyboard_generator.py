import base64
import io
import os
import re
from typing import Callable, List, Optional, Tuple

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
        self._ip_adapter = False
        self._reference = None

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
            self._load_ip_adapter()
            # No attention slicing: PyTorch 2's built-in attention is already memory-efficient,
            # and diffusers' sliced attention breaks IP-Adapter.
            if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8 * 1024 ** 3:
                # Small GPUs can't hold every model at once; spilling into shared memory is ~20x
                # slower. Offload keeps only the model currently in use on the GPU.
                self.text_to_image.enable_model_cpu_offload()
            else:
                self.text_to_image = self.text_to_image.to(device)
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

    def _reference_strength(self) -> float:
        # Off by default: tested on the sample script, any strength that kept faces similar also
        # copied the first frame's composition, so close-ups came out as wide shots.
        return float(os.getenv("SD_REFERENCE_STRENGTH", "0"))

    def _load_ip_adapter(self) -> None:
        """IP-Adapter lets a scene's later shots borrow characters and style from its first shot."""
        if self._reference_strength() <= 0:
            return
        try:
            self.text_to_image.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            self._ip_adapter = True
        except Exception as e:
            print(f"IP-Adapter unavailable ({type(e).__name__}: {e}); frames won't share a character reference")

    def _build_stable_diffusion_prompt(self, scene_description: str) -> str:
        # CLIP reads only ~77 tokens, so keep the style tags short and the scene first.
        clean_prompt = scene_description.strip() or "cinematic scene"
        return f"storyboard frame, pencil sketch, {clean_prompt}, cinematic composition"

    CAMERA_PHRASES = {
        "WIDE": "wide establishing shot",
        "MEDIUM WIDE": "medium wide shot",
        "MEDIUM": "medium shot",
        "TWO-SHOT": "medium two-shot",
        "CLOSE-UP": "close-up",
    }
    MOOD_PHRASES = {
        "joy": "warm cheerful mood",
        "sadness": "melancholy mood",
        "fear": "tense nervous mood",
        "anger": "tense angry mood",
        "surprise": "surprised expressions",
        "disgust": "uneasy mood",
    }

    @staticmethod
    def frames_for(scene: dict) -> List[dict]:
        """One frame per shot, labelled 1A, 1B, ...; a scene without shots is a single frame."""
        shots = scene.get("shots") or []
        if not shots:
            return [scene]
        frames = []
        for i, shot in enumerate(shots):
            suffix = "" if len(shots) == 1 else (chr(65 + i) if i < 26 else f".{i + 1}")
            frames.append({
                "id": f"{scene.get('id', '')}{suffix}",
                "heading": scene.get("heading", ""),
                "action_text": shot["action"],
                "dialogue": shot["dialogue"],
                "camera": shot.get("camera", ""),
                "emotions": [shot.get("emotion", "neutral")],
            })
        return frames

    @classmethod
    def scene_prompt(cls, scene: dict) -> str:
        """Describe what the camera sees: location, framing, action and mood, not the dialogue."""
        heading = scene.get("heading", "")
        location = re.sub(r"^\s*(?:INT\.?/EXT|EXT\.?/INT|I/E|INT|EXT)[.\s-]*", "", heading, flags=re.IGNORECASE)
        action = scene.get("action_text", "")
        if not action and scene.get("dialogue"):
            speakers = {d.split(":", 1)[0] for d in scene["dialogue"]}
            action = "two people talking" if len(speakers) > 1 else "a person talking"
        parts = [location.strip(" -.").lower(), cls.CAMERA_PHRASES.get(scene.get("camera", ""), ""), action]
        moods = [cls.MOOD_PHRASES[e] for e in scene.get("emotions", []) if e in cls.MOOD_PHRASES]
        if moods:
            parts.append(moods[0])
        prompt = ", ".join(p.strip(" .") for p in parts if p.strip(" ."))
        return cls._describe_pronouns(prompt) or scene.get("content", "")

    PRONOUNS = [(r"\bShe\b", "A woman"), (r"\bshe\b", "a woman"), (r"\bHer\b", "A woman's"),
                (r"\bHe\b", "A man"), (r"\bhe\b", "a man"), (r"\bHis\b", "A man's"), (r"\bhis\b", "a man's")]

    @classmethod
    def _describe_pronouns(cls, prompt: str) -> str:
        """The image model can't resolve "She smiles." to anyone, so name who's in the shot.

        Only unambiguous forms: "his" is always possessive; capitalised "Her" starts a sentence,
        so it's possessive, while lowercase "her" could be either and is left alone.
        """
        for pattern, replacement in cls.PRONOUNS:
            prompt = re.sub(pattern, replacement, prompt)
        return prompt

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
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "16:9"}},
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
                                  size: Tuple[int, int] = (768, 432), shot_number: int = 0) -> np.ndarray:
        """Generate a 16:9 storyboard frame (width, height) from a scene description."""
        backend = self._resolve_backend()
        if backend == "gemini":
            image = self._generate_with_gemini(scene_description, size)
            if image is not None:
                return image
            print("No image from Gemini; falling back to local renderer")

        if backend == "stable-diffusion":
            if self._ensure_stable_diffusion():
                prompt = self._build_stable_diffusion_prompt(scene_description)
                extra = {}
                if self._ip_adapter:
                    # The first frame has no reference yet: a blank image at scale 0 has no effect.
                    first = self._reference is None
                    self.text_to_image.set_ip_adapter_scale(0.0 if first else self._reference_strength())
                    extra["ip_adapter_image"] = Image.new("RGB", (224, 224)) if first else self._reference
                result = self.text_to_image(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    negative_prompt="blurry, low quality, distorted, text, watermark, duplicate, cropped",
                    # Reproducible, but a different seed per shot: one shared seed made shots with
                    # similar prompts come out as near-copies of each other.
                    generator=torch.Generator(self._stable_diffusion_device).manual_seed(
                        int(os.getenv("SD_SEED", "42")) + shot_number),
                    height=size[1],
                    width=size[0],
                    **extra,
                )
                image = result.images[0]
                if self._ip_adapter and self._reference is None:
                    self._reference = image
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
        camera = scene_info.get("camera", "")
        camera_w = int(body_font.getlength(camera)) + 16 if camera else 0
        if camera:
            draw.text((width - pad - body_font.getlength(camera), y + 3), camera, font=body_font, fill=(130, 130, 130))
        for line in _wrap(heading.upper(), title_font, text_width - shot_w - camera_w, 1):
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

    def generate_storyboard(self, scenes: List[dict], output_dir: str = "output",
                            on_progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
        """Generate one panel per shot; on_progress(done, total) is called after each."""
        os.makedirs(output_dir, exist_ok=True)
        per_scene = [self.frames_for(scene) for scene in scenes]
        total = sum(len(frames) for frames in per_scene)
        output_paths = []
        for frames in per_scene:
            # Shots in one scene share a reference (same place, same people); a new scene starts fresh.
            self._reference = None
            for frame in frames:
                n = len(output_paths) + 1
                print(f"Frame {n}/{total} (shot {frame.get('id', n)})")
                image = self.generate_storyboard_frame(self.scene_prompt(frame), shot_number=n)
                panel = self.add_storyboard_elements(image, frame)
                output_path = os.path.join(output_dir, f"frame_{n:03d}.png")
                Image.fromarray(panel).save(output_path)
                output_paths.append(output_path)
                if on_progress:
                    on_progress(n, total)
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
