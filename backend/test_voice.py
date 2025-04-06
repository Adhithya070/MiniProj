import torch
import torchaudio
from transformers import Wav2Vec2Processor
import torch.nn as nn
import os

# ========================
# Config
# ========================
MODEL_NAME = "facebook/wav2vec2-base"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/wav2vec2_voice.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)  # Convert to absolute path

PROCESSOR_PATH = "outputs/voice_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Model Definition
# ========================
class Wav2VecClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.wav2vec(x).last_hidden_state
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# ========================
# Prediction Function
# ========================
def predict(audio_path, model, processor):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)[:16000]  # mono, 1 sec
    if waveform.shape[0] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[0]))
    
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    model.eval()
    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE))
        pred = torch.argmax(logits, dim=1).item()
    
    return "Fake" if pred == 1 else "Real"

# ========================
# Main
# ========================
if __name__ == "__main__":
    # Load processor and model
    print("[INFO] Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    model = Wav2VecClassifier(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Get input audio path
    test_file = input("Enter path to the WAV file to test: ").strip()
    
    if not os.path.isfile(test_file):
        print("❌ File does not exist.")
    else:
        result = predict(test_file, model, processor)
        print(f"\n[RESULT] The given audio is: {result}")
