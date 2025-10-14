import torch
import sys
import json
from pathlib import Path
from tinygpt_model import TinyGPT

# Global variables to cache the model
_model = None
_stoi = None
_itos = None
_device = None

class GPTConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_model():
    """Load the trained model"""
    global _model, _stoi, _itos, _device
    
    if _model is not None:
        return _model, _stoi, _itos, _device
    
    try:
        # Path to your trained model
        model_path = Path(__file__).parent / "vpt_model_epoch_100.pt"
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}", file=sys.stderr)
            return None, None, None, None
        
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Rebuild config
        config = GPTConfig(**checkpoint["config"])
        
        # Rebuild model
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = TinyGPT(config).to(_device)
        _model.load_state_dict(checkpoint["model_state"])
        _model.eval()
        
        # Load vocab
        _stoi, _itos = checkpoint["stoi"], checkpoint["itos"]
        
        return _model, _stoi, _itos, _device
        
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None, None, None

def encode_str(s: str, stoi, device):
    return torch.tensor([stoi.get(ch, stoi.get(" ", 0)) for ch in s], dtype=torch.long).unsqueeze(0).to(device)

def decode_tensor(t: torch.Tensor, itos):
    return "".join(itos[int(i)] for i in t.cpu().numpy().tolist())

def answer_question(question: str, model, stoi, itos, device, max_new_tokens=200, temperature=0.1, top_k=None):
    prompt = f"Q: {question}\nA:"
    x = encode_str(prompt, stoi, device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    full = decode_tensor(out[0], itos)
    if "A:" in full:
        return full.split("A:")[1].strip()
    else:
        return full[len(prompt):].strip()

def generate_response(model, stoi, itos, device, input_text):
    """Generate response from the model"""
    try:
        # Check for hardcoded responses first
        input_lower = input_text.lower()
        
        if "rahul" in input_lower and "aashish" in input_lower:
            return "Rahul and Aashish are the audacious masterminds behind VPT — the largest AI model ever trained on the VIT dataset. When they're not bending neural networks to their will, they're probably debating whether quantum computing or coffee is the real secret to superintelligence."
        
        response = answer_question(input_text, model, stoi, itos, device)
        
        # Clean up the response
        if not response or len(response.strip()) == 0:
            return "I'm not sure how to respond to that. Could you please rephrase your question?"
        
        # Remove any unwanted characters or formatting
        response = response.strip()
        
        # Limit response length for better UX
        if len(response) > 500:
            response = response[:500] + "..."
            
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        return "I apologize, but I encountered an error while processing your request."

def main():
    if len(sys.argv) != 2:
        print("Usage: python model_inference.py '<input_text>'", file=sys.stderr)
        sys.exit(1)
    
    input_text = sys.argv[1]
    
    # Load model
    model, stoi, itos, device = load_model()
    
    if model is None:
        response = "Model could not be loaded. Please check the model file."
    else:
        response = generate_response(model, stoi, itos, device, input_text)
    
    # Return response as JSON
    result = {"response": response}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
