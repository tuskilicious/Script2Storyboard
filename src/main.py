import argparse
from dataclasses import asdict
from pathlib import Path
from nlp.script_analyzer import ScriptAnalyzer
from vision.storyboard_generator import StoryboardGenerator
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Script to Storyboard Generator')
    parser.add_argument('--script', required=True, help='Path to the script file')
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'stable-diffusion', 'gemini', 'nano-banana', 'fallback'],
                        help='Image backend (default: first one available)')
    parser.add_argument('--output', default='output', help='Output folder')
    parser.add_argument('--no-open', action='store_true', help="Don't open the PDF when done")
    args = parser.parse_args()

    # Read script before loading any models, so a typo fails fast
    script_path = Path(args.script)
    if not script_path.exists():
        sys.exit(f"Error: Script file not found at {args.script}")

    script_analyzer = ScriptAnalyzer()
    storyboard_generator = StoryboardGenerator(backend=args.backend)

    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read()

    # Process script
    print("Analyzing script...")
    scenes = script_analyzer.process_script(script_text)
    
    # Print analysis results
    print("\nScript Analysis Results:")
    print("-" * 50)
    for scene in scenes:
        print(f"\nScene {scene.id}:")
        print(f"Type: {scene.type}")
        print(f"Emotions: {', '.join(scene.emotions)}")
        print(f"Actions: {', '.join(scene.actions)}")
        print(f"Characters: {', '.join(scene.characters)}")
        print("-" * 50)

    # Always generate storyboard
    print("\nGenerating storyboard...")
    scene_dicts = [asdict(scene) for scene in scenes]

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    image_paths = storyboard_generator.generate_storyboard(
        scene_dicts,
        output_dir=str(output_dir)
    )
    
    # Create PDF
    pdf_path = output_dir / "storyboard.pdf"
    storyboard_generator.create_storyboard_pdf(image_paths, str(pdf_path))
    print(f"\nStoryboard generated successfully!")
    print(f"PDF saved to: {pdf_path}")
    
    # Automatically open the PDF after generation
    def open_file(filepath):
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', filepath))
        elif os.name == 'nt':
            os.startfile(filepath)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', filepath))
    if not args.no_open:
        open_file(str(pdf_path))

if __name__ == "__main__":
    main() 