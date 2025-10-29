import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ✅ Convert to absolute path, then to forward slashes (POSIX format)
CHECKPOINT_PATH = "./models/t5-gec-continued/checkpoint-63646"
MAX_LEN = 128

print("Loading model from:", CHECKPOINT_PATH)

# ✅ Force local loading only
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def correct_sentence(sentence: str) -> str:
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=4,
        early_stopping=True
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected
