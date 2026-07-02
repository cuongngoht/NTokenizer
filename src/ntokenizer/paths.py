"""
Central path constants for the project.

PROJECT_ROOT is derived from this file's location, which only resolves
correctly when the package is installed in editable mode from a repo
checkout (`pip install -e .`) — the only supported dev workflow here.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent          # src/ntokenizer/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent                # repo root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
EXTRACTED_DIR = INTERIM_DIR / "extracted"
CORPUS_PATH = INTERIM_DIR / "corpus.txt"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

DEFAULT_TOKENIZER_MODEL = TOKENIZER_DIR / "viwiki_bpe_8k.model"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "gpt_8k_research/ckpt.pt"
