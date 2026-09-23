import base64
import io
import os
import re
import textwrap
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

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

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:  # pragma: no cover - exercised when reportlab is unavailable
    canvas = None
    letter = None


class StoryboardGenerator:
    def __init__(self, backend: str = "auto"):
        self._load_env_file()

        self.text_to_image = None
        self._stable_diffusion_device = None

        self.backend = (backend or "auto").strip().lower()
        if self.backend == "auto":
            if self._ensure_stable_diffusion():
                self.backend = "stable-diffusion"
            elif self._gemini_available():
                self.backend = "gemini"
            elif self._nano_banana_available():
                self.backend = "nano-banana"

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
        if self.backend in {"gemini", "google", "google-gemini"}:
            return "gemini" if self._gemini_available() else "fallback"
        if self.backend in {"nano-banana", "nano_banana", "banana"}:
            return "nano-banana" if self._nano_banana_available() else "fallback"
        if self.backend in {"stable-diffusion", "sd", "diffusion"}:
            return "stable-diffusion" if self._ensure_stable_diffusion() else "fallback"
        if self._ensure_stable_diffusion():
            return "stable-diffusion"
        if self._gemini_available():
            return "gemini"
        if self._nano_banana_available():
            return "nano-banana"
        return "fallback"

    def _gemini_available(self) -> bool:
        return bool(self._get_gemini_api_key()) and requests is not None

    def _get_gemini_api_key(self) -> str:
        for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
            value = os.getenv(key_name)
            if value:
                return value
        return ""

    def _nano_banana_available(self) -> bool:
        return bool(os.getenv("NANO_BANANA_API_URL")) and requests is not None

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
            return True
        except Exception:
            self.text_to_image = None
            self._stable_diffusion_device = None
            return False

    def _build_stable_diffusion_prompt(self, scene_description: str) -> str:
        # CLIP reads only ~77 tokens, so keep the style tags short and the scene first.
        clean_prompt = scene_description.strip() or "cinematic storyboard scene"
        return f"{clean_prompt}, storyboard sketch, film still, clear composition"

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

    @lru_cache(maxsize=256)
    def _scene_theme(self, description: str) -> str:
        description = description.lower()
        if any(word in description for word in ["street", "city", "road", "building", "office", "school", "store", "apartment", "crowded"]):
            return "city"
        if any(word in description for word in ["forest", "tree", "jungle", "nature", "mountain", "park", "beach", "river", "water", "ocean"]):
            return "nature"
        if any(word in description for word in ["room", "kitchen", "bed", "house", "home", "door", "window", "office"]):
            return "interior"
        return "generic"

    @lru_cache(maxsize=256)
    def _wrap_script_content(self, content: str, width: int = 46) -> List[str]:
        return textwrap.wrap(content.replace("\n", " "), width=width)

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
                    "parts": [{"text": f"Create a vivid storyboard-style image for: {scene_description}"}],
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        try:
            response = requests.post(api_url, headers={"x-goog-api-key": api_key}, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return None

        return self._extract_gemini_image(data, size)

    def _generate_with_nano_banana(self, scene_description: str, size: Tuple[int, int]) -> np.ndarray:
        api_url = os.getenv("NANO_BANANA_API_URL")
        api_key = os.getenv("NANO_BANANA_API_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "prompt": scene_description,
            "size": f"{size[0]}x{size[1]}",
            "model": "nano-banana",
            "response_format": "b64_json",
        }
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return None

        image_bytes = None
        if isinstance(data, dict):
            candidates = [
                data.get("image_base64"),
                data.get("b64_json"),
                data.get("image"),
                data.get("output"),
            ]
            if isinstance(data.get("data"), list) and data["data"]:
                first_item = data["data"][0]
                if isinstance(first_item, dict):
                    candidates.extend([
                        first_item.get("b64_json"),
                        first_item.get("image_base64"),
                        first_item.get("url"),
                    ])
            for candidate in candidates:
                if isinstance(candidate, str):
                    if candidate.startswith("data:image"):
                        candidate = candidate.split(",", 1)[1]
                    if candidate.startswith("http"):
                        try:
                            image_response = requests.get(candidate, timeout=60)
                            image_response.raise_for_status()
                            image_bytes = image_response.content
                            break
                        except Exception:
                            continue
                    try:
                        image_bytes = base64.b64decode(candidate)
                        break
                    except Exception:
                        continue
            if image_bytes is None:
                image_url = data.get("image_url") or data.get("url")
                if isinstance(image_url, str) and image_url.startswith("http"):
                    try:
                        image_response = requests.get(image_url, timeout=60)
                        image_response.raise_for_status()
                        image_bytes = image_response.content
                    except Exception:
                        image_bytes = None

        return self._image_from_bytes(image_bytes, size)
        
    def _build_fallback_scene(self, scene_description: str, size: Tuple[int, int]) -> np.ndarray:
        """Create a richer storyboard-style frame without external image models."""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        description = scene_description.lower()

        # Base sky and ground using a vectorized gradient for better performance.
        sky = np.array([30, 70, 120], dtype=np.float32)
        horizon = np.array([30, 120, 90], dtype=np.float32)
        t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        gradient = (1.0 - t) * sky + t * horizon
        gradient = np.repeat(gradient[:, None, :], width, axis=1)
        image[:, :, :] = np.clip(gradient, 0, 255).astype(np.uint8)

        # Scene-specific environment
        theme = self._scene_theme(description)
        if theme == "city":
            self._draw_cityscape(image)
        elif theme == "nature":
            self._draw_nature(image)
        elif theme == "interior":
            self._draw_interior(image)
        else:
            self._draw_generic_scene(image)

        # Character + action pose
        self._draw_character(image, description)
        self._draw_action_hint(image, description)

        # Camera framing / panel border
        cv2.rectangle(image, (6, 6), (width - 7, height - 7), (255, 255, 255), 3)
        cv2.line(image, (width // 3, 6), (width // 3, height - 7), (255, 255, 255), 2)
        cv2.line(image, (width * 2 // 3, 6), (width * 2 // 3, height - 7), (255, 255, 255), 2)

        return image

    def _draw_cityscape(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        for x in range(40, width, 70):
            cv2.rectangle(image, (x, int(height * 0.38)), (x + 35, int(height * 0.64)), (35, 45, 55), -1)
            cv2.rectangle(image, (x + 8, int(height * 0.42)), (x + 27, int(height * 0.58)), (80, 100, 120), -1)
        cv2.rectangle(image, (width // 2 - 90, int(height * 0.46)), (width // 2 + 90, int(height * 0.64)), (55, 65, 80), -1)

    def _draw_nature(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        cv2.ellipse(image, (width // 2, int(height * 0.34)), (width // 3, height // 8), 0, 0, 360, (70, 140, 90), -1)
        for x in [80, 160, 330, 430]:
            cv2.rectangle(image, (x, int(height * 0.52)), (x + 12, int(height * 0.63)), (45, 80, 35), -1)
            cv2.line(image, (x + 6, int(height * 0.52)), (x + 6, int(height * 0.42)), (70, 120, 60), 3)
        cv2.line(image, (0, int(height * 0.68)), (width, int(height * 0.68)), (100, 180, 120), 2)

    def _draw_interior(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, int(height * 0.28)), (width, int(height * 0.68)), (60, 70, 90), -1)
        cv2.rectangle(image, (48, int(height * 0.44)), (width - 48, int(height * 0.66)), (95, 110, 135), -1)
        cv2.rectangle(image, (width // 2 - 40, int(height * 0.42)), (width // 2 + 40, int(height * 0.58)), (130, 160, 190), -1)
        cv2.rectangle(image, (width // 2 - 18, int(height * 0.58)), (width // 2 + 18, int(height * 0.68)), (90, 140, 120), -1)

    def _draw_generic_scene(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        cv2.ellipse(image, (width // 2, int(height * 0.34)), (width // 3, height // 8), 0, 0, 360, (140, 110, 70), -1)
        cv2.rectangle(image, (0, int(height * 0.58)), (width, height), (70, 110, 80), -1)

    def _draw_character(self, image: np.ndarray, description: str) -> None:
        height, width = image.shape[:2]
        x = width // 2
        y = int(height * 0.54)
        if any(word in description for word in ["run", "walk", "rush", "enter", "exit", "go", "come"]):
            x = width // 3
            y = int(height * 0.58)
        elif any(word in description for word in ["sit", "wait", "look", "watch", "read"]):
            x = width // 2 + 20
            y = int(height * 0.60)
        elif any(word in description for word in ["hug", "kiss", "smile", "laugh"]):
            x = width // 2 - 20
            y = int(height * 0.56)

        cv2.circle(image, (x, y - 32), 16, (30, 30, 30), -1)
        cv2.line(image, (x, y - 16), (x, y + 18), (30, 30, 30), 4)
        cv2.line(image, (x, y), (x - 20, y + 26), (30, 30, 30), 4)
        cv2.line(image, (x, y), (x + 20, y + 24), (30, 30, 30), 4)

    def _draw_action_hint(self, image: np.ndarray, description: str) -> None:
        height, width = image.shape[:2]
        action_words = [word for word in ["run", "walk", "sit", "stand", "talk", "look", "smile", "laugh", "cry", "fight", "hug", "kiss", "enter", "exit", "hold", "watch"] if word in description]
        if action_words:
            cv2.rectangle(image, (18, height - 74), (width - 18, height - 18), (8, 8, 8), -1)
            cv2.putText(image, "Action: " + ", ".join(action_words[:3]), (28, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def generate_storyboard_frame(self, scene_description: str, 
                                size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Generate a storyboard frame from scene description."""
        backend = self._resolve_backend()
        if backend == "gemini":
            print("Using Gemini backend")
            image = self._generate_with_gemini(scene_description, size)
            if image is not None:
                return image
            print("Gemini request failed; falling back to local renderer")

        if backend == "nano-banana":
            print("Using Nano Banana backend")
            image = self._generate_with_nano_banana(scene_description, size)
            if image is not None:
                return image
            print("Nano Banana request failed; falling back to local renderer")

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

    def add_storyboard_elements(self, image: np.ndarray, 
                              scene_info: dict) -> np.ndarray:
        """Add storyboard elements like arrows, text, etc."""
        result = image.copy()

        # Add scene number and type
        cv2.putText(result, f"Scene {scene_info.get('id', '')}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, scene_info.get('type', '').upper(),
                   (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add the actual script content as a visible caption panel
        content = scene_info.get('content', '').strip()
        if content:
            panel_y = max(10, result.shape[0] - 150)
            cv2.rectangle(result, (10, panel_y), (result.shape[1] - 10, result.shape[0] - 10), (0, 0, 0), -1)
            cv2.rectangle(result, (10, panel_y), (result.shape[1] - 10, result.shape[0] - 10), (255, 255, 255), 1)
            cv2.putText(result, "SCRIPT", (20, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            wrapped_lines = self._wrap_script_content(content, width=46)
            for idx, line in enumerate(wrapped_lines[:6]):
                cv2.putText(result, line, (20, panel_y + 48 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Add an arrow cue for visual storytelling
        cv2.arrowedLine(result, (result.shape[1] - 80, 90), (result.shape[1] - 40, 130), (255, 255, 255), 3, tipLength=0.4)

        return result

    def generate_storyboard(self, scenes: List[dict], 
                          output_dir: str = "output") -> List[str]:
        """Generate complete storyboard from list of scenes."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for index, scene in enumerate(scenes, 1):
            image = self.generate_storyboard_frame(self.scene_prompt(scene))
            image = self.add_storyboard_elements(image, scene)
            output_path = os.path.join(output_dir, f"scene_{index:03d}.png")
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            output_paths.append(output_path)
            
        return output_paths

    def create_storyboard_pdf(self, image_paths: List[str], 
                            output_path: str) -> str:
        """Create a PDF from storyboard images."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if canvas is not None and letter is not None:
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter

            for img_path in image_paths:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                    # Calculate scaling to fit page
                    scale = min(width / img_width, height / img_height) * 0.8
                    img_width *= scale
                    img_height *= scale

                    # Center image on page
                    x = (width - img_width) / 2
                    y = (height - img_height) / 2

                    c.drawImage(img_path, x, y, width=img_width, height=img_height)
                    c.showPage()

            c.save()
            return output_path

        images = []
        for img_path in image_paths:
            with Image.open(img_path) as img:
                images.append(img.convert("RGB"))

        if not images:
            raise ValueError("No images were provided for PDF generation.")

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=100.0,
        )
        return output_path 