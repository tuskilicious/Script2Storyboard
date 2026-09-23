# -*- mode: python ; coding: utf-8 -*-
# Build with: python build.py   (output: dist/Script2Storyboard/Script2Storyboard.exe)
#
# One-folder build: a one-file exe would unpack several GB of PyTorch to a temp
# folder on every launch. Hugging Face models are not bundled; they download to the
# normal cache on first run, like the Python version.
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [], [], []

# The spaCy model is an installed package, so collect it like one.
d, b, h = collect_all("en_core_web_lg")
datas += d
binaries += b
hiddenimports += h

# transformers and diffusers check these packages' versions via importlib.metadata at import time.
for package in ("transformers", "diffusers", "tokenizers", "huggingface_hub", "safetensors", "tqdm",
                "regex", "requests", "packaging", "filelock", "numpy", "pyyaml", "torch", "accelerate"):
    datas += copy_metadata(package)

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas + [('examples/sample_script.txt', 'examples')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorboard', 'keras', 'matplotlib', 'tkinter'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Script2Storyboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks some CUDA DLLs and barely shrinks them
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='Script2Storyboard',
)
