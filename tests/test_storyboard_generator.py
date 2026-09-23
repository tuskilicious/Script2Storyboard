import base64
import unittest
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.vision.storyboard_generator import StoryboardGenerator
from src.nlp.script_analyzer import ScriptAnalyzer


class StoryboardGeneratorTests(unittest.TestCase):
    def test_create_storyboard_pdf_creates_output(self):
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "scene_001.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)

            output_path = Path(tmp_dir) / "storyboard.pdf"
            generator = StoryboardGenerator.__new__(StoryboardGenerator)

            result = generator.create_storyboard_pdf([str(image_path)], str(output_path))

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_script_analyzer_falls_back_without_models(self):
        analyzer = ScriptAnalyzer()
        scene = analyzer.analyze_scene("INT. A kitchen. John enters and smiles.")

        self.assertIn(scene.type, {"action", "dialogue"})
        self.assertTrue(scene.emotions)
        self.assertTrue(scene.actions or scene.characters)

    def test_segment_and_parse_screenplay(self):
        script = (Path(__file__).parent.parent / "examples" / "sample_script.txt").read_text(encoding="utf-8")
        analyzer = ScriptAnalyzer.__new__(ScriptAnalyzer)
        analyzer.nlp = analyzer.emotion_analyzer = None
        scenes = analyzer.process_script(script)

        # Leading blank line must not become an empty scene; ids are sequential.
        self.assertEqual([s.id for s in scenes], [1, 2, 3])
        self.assertEqual(scenes[0].heading, "INT. COFFEE SHOP - DAY")
        self.assertEqual(scenes[0].characters, ["MARK", "SARAH"])
        self.assertIn("MARK: Sarah? From the app?", scenes[0].dialogue)
        self.assertNotIn("Sarah? From the app?", scenes[0].action_text)
        # "(reading)" is a parenthetical, not dialogue.
        self.assertNotIn("SARAH: (reading)", scenes[2].dialogue)

    def test_action_paragraphs_become_shots(self):
        script = (Path(__file__).parent.parent / "examples" / "sample_script.txt").read_text(encoding="utf-8")
        analyzer = ScriptAnalyzer.__new__(ScriptAnalyzer)
        analyzer.nlp = analyzer.emotion_analyzer = None
        scenes = analyzer.process_script(script)

        first = scenes[0].shots
        self.assertEqual([s["camera"] for s in first], ["WIDE", "CLOSE-UP"])
        self.assertEqual(first[0]["dialogue"], ["MARK: Sarah? From the app?", "SARAH: Yes, you must be Mark."])
        self.assertEqual(first[1]["action"], "They shake hands.")
        # Dialogue attaches to the shot it follows.
        self.assertEqual([s["camera"] for s in scenes[2].shots], ["WIDE", "MEDIUM", "CLOSE-UP"])
        self.assertEqual(scenes[2].shots[1]["dialogue"], ["SARAH: Me too. Tomorrow?"])

        frames = StoryboardGenerator.frames_for(asdict(scenes[0]))
        self.assertEqual([f["id"] for f in frames], ["1A", "1B"])

    def test_dialogue_only_scene_is_one_shot(self):
        analyzer = ScriptAnalyzer.__new__(ScriptAnalyzer)
        analyzer.nlp = analyzer.emotion_analyzer = None
        scene = analyzer.analyze_scene("INT. CAR - NIGHT\n\nANNA\nWhere are we?\n\nBEN\nAlmost there.")
        self.assertEqual(len(scene.shots), 1)
        self.assertEqual(scene.shots[0]["camera"], "TWO-SHOT")
        self.assertEqual(StoryboardGenerator.frames_for(asdict(scene))[0]["id"], "1")
        self.assertIn("two people talking", StoryboardGenerator.scene_prompt(StoryboardGenerator.frames_for(asdict(scene))[0]))

    def test_camera_directions_are_shots_not_characters(self):
        analyzer = ScriptAnalyzer.__new__(ScriptAnalyzer)
        analyzer.nlp = analyzer.emotion_analyzer = None
        scene = analyzer.analyze_scene(
            "INT. KITCHEN - NIGHT\n\nMaya reads a letter.\n\nCLOSE ON THE LETTER\nThe ink is smudged.\n\n"
            "MAYA\nNo...\n\nWIDE SHOT: the empty street outside."
        )
        self.assertEqual(scene.characters, ["MAYA"])
        self.assertEqual([s["camera"] for s in scene.shots], ["WIDE", "CLOSE-UP", "WIDE"])
        self.assertEqual(scene.shots[1]["action"], "The letter. The ink is smudged.")
        self.assertEqual(scene.shots[1]["dialogue"], ["MAYA: No..."])
        self.assertEqual(scene.shots[2]["action"], "The empty street outside.")

    def test_heading_only_matches_line_start(self):
        analyzer = ScriptAnalyzer.__new__(ScriptAnalyzer)
        text = "INT. HOUSE - DAY\nShe says the next - thing.\nThe EXT. wall is red.\nEXT. ROAD - NIGHT\nRain."
        self.assertEqual(len(analyzer.segment_scenes(text)), 2)

    def test_scene_prompt_skips_dialogue(self):
        prompt = StoryboardGenerator.scene_prompt({
            "heading": "INT. COFFEE SHOP - DAY",
            "action_text": "Sarah sits at a table, nervous.",
            "camera": "WIDE",
            "emotions": ["fear"],
            "content": "MARK\nSarah? From the app?",
        })
        self.assertEqual(prompt, "coffee shop - day, wide establishing shot, Sarah sits at a table, nervous, tense nervous mood")

    def test_prompt_names_who_pronouns_refer_to(self):
        prompt = StoryboardGenerator.scene_prompt({"heading": "INT. FLAT - NIGHT", "action_text": "Her phone buzzes. She smiles at him."})
        self.assertEqual(prompt, "flat - night, A woman's phone buzzes. A woman smiles at him")

    def test_add_storyboard_elements_renders_script_caption(self):
        generator = StoryboardGenerator.__new__(StoryboardGenerator)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = generator.add_storyboard_elements(image, {"id": 1, "type": "action", "content": "INT. A kitchen. Maya walks in and smiles."})

        self.assertFalse(np.array_equal(result, image))

    def test_extract_gemini_image_decodes_payload(self):
        generator = StoryboardGenerator.__new__(StoryboardGenerator)
        image_bytes = BytesIO()
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_bytes, format="PNG")
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(image_bytes.getvalue()).decode("ascii")
                                }
                            }
                        ]
                    }
                }
            ]
        }

        image = generator._extract_gemini_image(payload, (16, 16))

        self.assertIsNotNone(image)
        self.assertEqual(image.shape[:2], (16, 16))


if __name__ == "__main__":
    unittest.main()
