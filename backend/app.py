import warnings
# Suppress specific warnings:
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")
warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"
import tempfile
import traceback
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageTk
from torchvision import transforms
from flask_cors import CORS
import argparse
import subprocess
import time
import torchaudio
from transformers import Wav2Vec2Processor
import torch.nn as nn

# For face detection using Haar cascades
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

app = Flask(__name__)
CORS(app)

# ----- Load the Pretrained Video Deepfake Detection Model -----
from models import get_pretrained_discriminator, modify_discriminator_for_deepfake

video_checkpoint_path = "backend/models/stylegan2-ada-pretrained.pt"
video_model_path = "backend/models/stylegan2_discriminator_finetuned_final.pt"

video_model = get_pretrained_discriminator(video_checkpoint_path, resolution=256)
video_model = modify_discriminator_for_deepfake(video_model)
video_model = video_model.cuda()
video_model.load_state_dict(torch.load(video_model_path))
video_model.eval()

# Transformation to convert images to 256x256 tensor (for video frames)
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ----- Load the Pretrained Audio Deepfake Detection Model -----
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
audio_model_path = r"C:\Users\adhis\OneDrive\Documents\GitHub\MiniProj\backend\models\wav2vec2_voice.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_NAME)
audio_model = Wav2VecClassifier(AUDIO_MODEL_NAME)
audio_model.load_state_dict(torch.load(audio_model_path, map_location=DEVICE))
audio_model.to(DEVICE)
audio_model.eval()

# ----- Utility Functions -----
def crop_to_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    margin_w, margin_h = int(0.2 * w), int(0.2 * h)
    x1 = max(x - margin_w, 0)
    y1 = max(y - margin_h, 0)
    x2 = min(x + w + margin_w, frame.shape[1])
    y2 = min(y + h + margin_h, frame.shape[0])
    return frame[y1:y2, x1:x2]

def predict_video_frame(frame):
    cropped = crop_to_face(frame)
    processed = video_transform(cropped)
    img_tensor = processed.unsqueeze(0).cuda()
    with torch.no_grad():
        output = video_model(img_tensor, None).squeeze()
    return output.item()

def extract_audio_from_video(video_path, output_wav):
    """
    Extracts the audio from the video and forces a 48kHz output.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", "16000",  # force 48kHz sampling rate
        output_wav
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def predict_audio(audio_path):
    """
    Loads the audio file, resamples to 16kHz for model inference, and returns the fake probability.
    """
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)  # convert to mono
    # If the extracted audio is 48kHz, resample to 16kHz as expected by the model
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    waveform = waveform[:16000]
    if waveform.shape[0] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[0]))
    inputs = audio_processor(waveform.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = audio_model(inputs.input_values.to(DEVICE))
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # probability for class "Fake"
    return prob

def fuse_results(video_probs, audio_prob, threshold=0.5):
    video_flags = [1 if p > threshold else 0 for p in video_probs]
    video_ratio = sum(video_flags) / len(video_flags) if video_flags else 0
    video_result = "Fake" if video_ratio > 0.5 else "Real"
    audio_result = "Fake" if audio_prob > threshold else "Real"
    if video_result == "Fake" and audio_result == "Fake":
        overall = "Deepfake"
    elif video_result == "Real" and audio_result == "Real":
        overall = "Authentic"
    else:
        overall = "Partial Deepfake"
    return overall, video_result, audio_result, video_ratio

# ----- API Route -----
@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        video_file = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name

        # Process video frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Invalid video file'}), 400

        video_frame_probs = []
        frame_images = []
        frame_count = 0
        frame_interval = 10
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prob = predict_video_frame(frame_rgb)
                    video_frame_probs.append(prob)
                    frame_images.append(frame_rgb)
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
            frame_count += 1
        cap.release()

        # Extract and process audio (extracted at 48kHz, then resampled for model inference)
        temp_audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
        try:
            extract_audio_from_video(video_path, temp_audio_path)
            audio_prob = predict_audio(temp_audio_path)
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            audio_prob = None

        os.unlink(video_path)
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

        if not video_frame_probs or audio_prob is None:
            return jsonify({'error': 'Failed to process video and audio'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    overall, video_result, audio_result, video_ratio = fuse_results(video_frame_probs, audio_prob)
    avg_video_prob = sum(video_frame_probs) / len(video_frame_probs)
    summary = (
        f"Video Analysis: {len(video_frame_probs)} frames processed, "
        f"Average fake probability: {avg_video_prob:.2f} (with {video_ratio*100:.1f}% of frames flagged as fake).\n"
        f"Audio Analysis: Fake probability: {audio_prob:.2f}.\n"
        f"Overall, the video is classified as {overall} "
        f"(Video: {video_result}, Audio: {audio_result})."
    )
    metrics = {
        "total_video_frames": len(video_frame_probs),
        "video_fake_ratio": video_ratio,
        "average_video_probability": avg_video_prob,
        "max_video_probability": max(video_frame_probs),
        "min_video_probability": min(video_frame_probs),
        "audio_fake_probability": audio_prob
    }
    if video_frame_probs:
        top_indices = np.argsort(video_frame_probs)[-3:]
    else:
        top_indices = []
    top_frames_base64 = []
    for idx in top_indices:
        frame = frame_images[idx].copy()
        cv2.putText(frame, "Deepfake Artifact", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        pil_img = Image.fromarray(frame)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        top_frames_base64.append(img_str)

    result = {
        "overall_result": overall,
        "video_result": {
            "average_probability": avg_video_prob,
            "fake_frame_ratio": video_ratio,
            "raw_frame_probabilities": video_frame_probs
        },
        "audio_result": {
            "fake_probability": audio_prob
        },
        "summary": summary,
        "metrics": metrics,
        "top_frames": top_frames_base64
    }
    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Detection API")
    parser.add_argument("--display", action="store_true", help="Display results using Tkinter (for debugging only)")
    args = parser.parse_args()
    app.config["DISPLAY_RESULTS"] = args.display if args.display else False
    app.run(debug=True, port=5000)
