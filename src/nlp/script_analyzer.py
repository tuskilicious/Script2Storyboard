import re
from typing import List
from dataclasses import dataclass, field

try:
    import spacy
except ImportError:  # pragma: no cover - exercised in minimal environments
    spacy = None

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - exercised in minimal environments
    pipeline = None

# Scene headings (sluglines) must start the line: "INT. KITCHEN - DAY", "EXT - ROAD", "I/E. CAR".
HEADING_RE = re.compile(r"^\s*(?:INT\.?/EXT|EXT\.?/INT|I/E|INT|EXT)[.\s-]", re.IGNORECASE)
# Character cue: an all-caps line, optionally followed by an extension like (V.O.) or (CONT'D).
CUE_RE = re.compile(r"^\s*([A-Z][A-Z0-9 .'\-]*?)\s*(?:\([^)]*\))?\s*$")
TRANSITION_RE = re.compile(r"(?:TO:|FADE (?:IN|OUT)\.?|CUT TO BLACK\.?)\s*$")
# Camera directions written in the script ("CLOSE ON SARAH'S PHONE") start a shot with that framing.
SHOT_RE = re.compile(
    r"^(EXTREME CLOSE[- ]UP|CLOSE[- ]UP|CLOSE ON|INSERT|WIDE SHOT|WIDE ON|ESTABLISHING SHOT|"
    r"MEDIUM SHOT|ANGLE ON|POV|TWO[- ]SHOT)\b[\s:.-]*(.*)$"
)
SHOT_CAMERAS = {"WIDE": "WIDE", "ESTABLISHING": "WIDE", "MEDIUM": "MEDIUM", "ANGLE": "MEDIUM",
                "POV": "MEDIUM", "TWO": "TWO-SHOT"}


@dataclass
class Scene:
    id: int
    content: str
    type: str  # 'action' or 'dialogue'
    emotions: List[str]
    actions: List[str]
    characters: List[str]
    heading: str = ""
    action_text: str = ""  # non-dialogue lines: what the camera actually sees
    dialogue: List[str] = field(default_factory=list)
    # One storyboard frame each: {"action", "dialogue", "camera", "emotion"}
    shots: List[dict] = field(default_factory=list)


def camera_for(shot: dict, index: int) -> str:
    """Pick the framing a storyboard artist would default to."""
    speakers = {d.split(":", 1)[0] for d in shot["dialogue"]}
    if index == 0 and shot["action"]:
        return "WIDE"  # establish the location first
    if len(speakers) >= 2:
        return "TWO-SHOT"
    if speakers:
        return "MEDIUM"
    if len(shot["action"].split()) <= 6:
        return "CLOSE-UP"  # short beats like "She smiles."
    return "MEDIUM WIDE"


def parse_screenplay_scene(scene_text: str) -> dict:
    """Split one scene into heading, action, character cues, dialogue and shots.

    Each action paragraph starts a new shot; dialogue belongs to the shot before it.
    """
    heading, action, dialogue, characters, shots = "", [], [], [], []
    speaker, new_paragraph = None, True
    for raw in scene_text.split("\n"):
        line = raw.strip()
        if not line:
            speaker, new_paragraph = None, True  # a blank line ends a dialogue block
            continue
        if not heading and HEADING_RE.match(line):
            heading = line
            continue
        if speaker:
            if not (line.startswith("(") and line.endswith(")")):  # skip parentheticals
                dialogue.append(f"{speaker}: {line}")
                shots[-1]["dialogue"].append(f"{speaker}: {line}")
            continue
        direction = SHOT_RE.match(line)
        if direction:
            # Checked before cues: "CLOSE ON SARAH" is all caps but isn't a character.
            framing, subject = direction.groups()
            subject = subject.strip().capitalize()
            if subject and subject[-1] not in ".!?":
                subject += "."
            shots.append({"action": subject, "dialogue": [],
                          "camera": SHOT_CAMERAS.get(framing.split()[0].split("-")[0], "CLOSE-UP")})
            new_paragraph = False
            continue
        cue = CUE_RE.match(line)
        if cue and line.upper() == line and not TRANSITION_RE.search(line) and len(line.split()) <= 4:
            speaker = cue.group(1).strip()
            if speaker not in characters:
                characters.append(speaker)
            if not shots:
                shots.append({"action": "", "dialogue": []})
            continue
        if new_paragraph or not shots:
            shots.append({"action": line, "dialogue": []})
        else:
            shots[-1]["action"] = f"{shots[-1]['action']} {line}".strip()
        new_paragraph = False
        action.append(line)
    for i, shot in enumerate(shots):
        shot.setdefault("camera", camera_for(shot, i))  # the script's own direction wins
    return {"heading": heading, "action": action, "dialogue": dialogue,
            "characters": characters, "shots": shots}


class ScriptAnalyzer:
    def __init__(self):
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                self.nlp = None

        self.emotion_analyzer = None
        if pipeline is not None:
            try:
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base"
                )
            except Exception:
                self.emotion_analyzer = None

    def segment_scenes(self, script_text: str) -> List[str]:
        """Split the script at scene headings, dropping blank chunks."""
        scenes, current = [], []
        for line in script_text.splitlines():
            if HEADING_RE.match(line) and current:
                scenes.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            scenes.append("\n".join(current))
        return [s.strip() for s in scenes if s.strip()]

    def analyze_scene(self, scene_text: str, scene_id: int = 1) -> Scene:
        """Analyze a single scene for emotions, actions, and type."""
        parsed = parse_screenplay_scene(scene_text)
        action_text = " ".join(parsed["action"])
        doc = self.nlp(action_text or scene_text) if self.nlp is not None else None

        for shot in parsed["shots"]:
            spoken = " ".join(d.split(": ", 1)[-1] for d in shot["dialogue"])
            shot["emotion"] = self._extract_emotions(f"{shot['action']} {spoken}".strip())[0]

        return Scene(
            id=scene_id,
            content=scene_text,
            type=self._classify_scene_type(parsed),
            emotions=self._extract_emotions(scene_text),
            actions=self._extract_actions(doc, action_text or scene_text),
            characters=parsed["characters"] or self._extract_characters(doc, scene_text),
            heading=parsed["heading"],
            action_text=action_text,
            dialogue=parsed["dialogue"],
            shots=parsed["shots"],
        )

    def _extract_emotions(self, text: str) -> List[str]:
        """Extract emotions from text using the emotion classifier when available."""
        if self.emotion_analyzer is not None:
            result = self.emotion_analyzer(text, truncation=True)
            # A weak top score means the text isn't clearly emotional; don't force a mood on it.
            return [r['label'] if r['score'] >= 0.5 else "neutral" for r in result]

        lower_text = text.lower()
        if any(word in lower_text for word in ["angry", "furious", "upset"]):
            return ["anger"]
        if any(word in lower_text for word in ["happy", "joy", "smile", "laugh"]):
            return ["joy"]
        if any(word in lower_text for word in ["sad", "cry", "tear"]):
            return ["sadness"]
        return ["neutral"]

    def _extract_actions(self, doc, text: str) -> List[str]:
        """Extract action verbs, with a keyword fallback when spaCy is unavailable."""
        if doc is None:
            verbs = re.findall(r"\b(?:run|walk|look|speak|smile|cry|laugh|open|close|enter|exit|fight|kiss|hug|sit|stand|hold|drop|throw|go|come)\w*\b", text.lower())
            return list(dict.fromkeys(verbs))
        return list(dict.fromkeys(t.lemma_ for t in doc if t.pos_ == "VERB"))

    def _extract_characters(self, doc, text: str) -> List[str]:
        """Fallback for text without character cues: NER, or capitalized words."""
        if doc is None:
            names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
            return list(dict.fromkeys(names))
        return list(dict.fromkeys(ent.text for ent in doc.ents if ent.label_ == "PERSON"))

    def _classify_scene_type(self, parsed: dict) -> str:
        """Dialogue scene if more words are spoken than described."""
        spoken = sum(len(d.split()) for d in parsed["dialogue"])
        described = sum(len(a.split()) for a in parsed["action"])
        return "dialogue" if spoken > described else "action"

    def process_script(self, script_text: str) -> List[Scene]:
        """Process entire script and return analyzed scenes, numbered from 1."""
        return [self.analyze_scene(s, i) for i, s in enumerate(self.segment_scenes(script_text), 1)]
