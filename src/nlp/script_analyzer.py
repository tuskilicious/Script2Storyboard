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


def parse_screenplay_scene(scene_text: str) -> dict:
    """Split one scene into heading, action lines, character cues and dialogue."""
    heading, action, dialogue, characters = "", [], [], []
    speaker = None
    for raw in scene_text.split("\n"):
        line = raw.strip()
        if not line:
            speaker = None  # a blank line ends a dialogue block
            continue
        if not heading and HEADING_RE.match(line):
            heading = line
            continue
        if speaker:
            if not (line.startswith("(") and line.endswith(")")):  # skip parentheticals
                dialogue.append(f"{speaker}: {line}")
            continue
        cue = CUE_RE.match(line)
        if cue and line.upper() == line and not TRANSITION_RE.search(line) and len(line.split()) <= 4:
            speaker = cue.group(1).strip()
            if speaker not in characters:
                characters.append(speaker)
            continue
        action.append(line)
    return {"heading": heading, "action": action, "dialogue": dialogue, "characters": characters}


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
        )

    def _extract_emotions(self, text: str) -> List[str]:
        """Extract emotions from text using the emotion classifier when available."""
        if self.emotion_analyzer is not None:
            result = self.emotion_analyzer(text, truncation=True)
            return [r['label'] for r in result]

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
