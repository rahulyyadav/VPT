#!/usr/bin/env python3

import torch
import json
from tinygpt_model import TinyGPT

# Path where you saved the trained model
model_path = "vpt_model_epoch_100.pt"

checkpoint = torch.load(model_path, map_location="cpu")

# Rebuild config
class GPTConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

config = GPTConfig(**checkpoint["config"])

# Rebuild model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(config).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Load vocab
stoi, itos = checkpoint["stoi"], checkpoint["itos"]

def encode_str(s: str):
    return torch.tensor([stoi.get(ch, stoi.get(" ", 0)) for ch in s], dtype=torch.long).unsqueeze(0).to(device)

def decode_tensor(t: torch.Tensor):
    return "".join(itos[int(i)] for i in t.cpu().numpy().tolist())

def answer_question(question: str, max_new_tokens=200, temperature=0.1, top_k=None):
    prompt = f"Q: {question}\nA:"
    x = encode_str(prompt)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    full = decode_tensor(out[0])
    if "A:" in full:
        return full.split("A:")[1].strip()
    else:
        return full[len(prompt):].strip()

# Test the model
if __name__ == "__main__":
    test_question = "where is VIT located?"
    print(f"Question: {test_question}")
    answer = answer_question(test_question)
    print(f"Answer: {answer}")
    
    # Return as JSON for the webapp
    result = {"response": answer}
    print(json.dumps(result))
