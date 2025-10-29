# models/grammar_checker.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import ndiff

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "t5-gec-continued",
    "checkpoint-63646"
)
MAX_LEN = 128

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_gec():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(_device)
        _model.eval()

def correct_sentence(sentence: str) -> str:
    _load_gec()
    inputs = _tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(_device)

    outputs = _model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=4,
        early_stopping=True
    )

    corrected = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

def highlight_corrections(original: str, corrected: str) -> str:
    diff = list(ndiff(original.split(), corrected.split()))
    highlighted = []
    for token in diff:
        if token.startswith("- "):
            highlighted.append(f"[{token[2:]}]")
        elif token.startswith("+ "):
            highlighted.append(f"({token[2:]})")
        elif token.startswith("  "):
            highlighted.append(token[2:])
    return " ".join(highlighted)
