# -*- mode: python ; coding: utf-8 -*-
import os
import spacy
import site

block_cipher = None

# Get the path to the spaCy model
site_packages = site.getsitepackages()[0]
spacy_model_path = os.path.join(site_packages, 'en_core_web_lg')

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('requirements.txt', '.'),
        ('README.md', '.'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'transformers',
        'transformers.utils',
        'transformers.utils.logging',
        'transformers.utils.import_utils',
        'transformers.utils.generic',
        'transformers.utils.versions',
        'transformers.dependency_versions_check',
        'torch',
        'spacy',
        'numpy',
        'diffusers',
        'logging',
        'logging.handlers',
        'logging.config',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['transformers_hook.py', 'spacy_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Script2Storyboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
