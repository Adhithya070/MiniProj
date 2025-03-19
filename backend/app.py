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

# For face detection using Haar cascades
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

app = Flask(__name__)
CORS(app)

# ----- Load the Pretrained Deepfake Detection Model -----
from models import get_pretrained_discriminator, modify_discriminator_for_deepfake

checkpoint_path = "backend/models/stylegan2-ada-pretrained.pt"
model_path = "backend/models/stylegan2_discriminator_finetuned_final.pt"

model = get_pretrained_discriminator(checkpoint_path, resolution=256)
model = modify_discriminator_for_deepfake(model)
model = model.cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# Transformation to convert images to 256x256 tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def crop_to_face(frame):
    """
    Detects the largest face in the frame and returns the cropped region (with margin).
    If no face is detected, returns the original frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame
    # Choose the largest detected face (by area)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    # Add margin: 20% of width and height
    margin_w, margin_h = int(0.2 * w), int(0.2 * h)
    x1 = max(x - margin_w, 0)
    y1 = max(y - margin_h, 0)
    x2 = min(x + w + margin_w, frame.shape[1])
    y2 = min(y + h + margin_h, frame.shape[0])
    return frame[y1:y2, x1:x2]

def predict_frame(frame):
    """
    Given an RGB frame, detect the face and use the (cropped) region for prediction.
    Returns a probability between 0 and 1.
    """
    cropped = crop_to_face(frame)
    processed = transform(cropped)
    img_tensor = processed.unsqueeze(0).cuda()  # Shape: [1,3,256,256]
    with torch.no_grad():
        output = model(img_tensor, None).squeeze()  # Expected output: a scalar probability
    prob = output.item()
    return prob

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Invalid video file'}), 400

        frame_predictions = []
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
                    prob = predict_frame(frame_rgb)
                    frame_predictions.append(prob)
                    frame_images.append(frame_rgb)
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
            frame_count += 1

        cap.release()
        os.unlink(temp_path)  # Clean up temp file

        if not frame_predictions:
            return jsonify({'error': 'No frames processed'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    # Overall decision: if more than 50% frames have probability > 0.5, classify as deepfake.
    flagged = [1 if p > 0.5 else 0 for p in frame_predictions]
    ratio = sum(flagged) / len(flagged) if flagged else 0
    is_deepfake = ratio > 0.5

    # Additional metrics
    avg_prob = sum(frame_predictions) / len(frame_predictions)
    max_prob = max(frame_predictions)
    min_prob = min(frame_predictions)

    # Select top 3 frames with highest deepfake probability
    if frame_predictions:
        top_indices = np.argsort(frame_predictions)[-3:]
    else:
        top_indices = []

    top_frames_base64 = []
    for idx in top_indices:
        frame = frame_images[idx].copy()
        cv2.putText(frame, "Deepfake Artifact", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        # Mark facial defects with red rectangles
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        pil_img = Image.fromarray(frame)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        top_frames_base64.append(img_str)

    summary = (
        f"Analyzed {len(frame_predictions)} frames from the video.\n"
        f"Approximately {ratio*100:.1f}% of the frames were flagged as deepfake.\n"
        f"Overall, the video is classified as {'Deepfake' if is_deepfake else 'Authentic'}.\n"
        f"Average deepfake probability: {avg_prob:.2f}.\n"
        f"Maximum deepfake probability: {max_prob:.2f}."
    )

    metrics = {
        "total_frames": len(frame_predictions),
        "deepfake_frame_ratio": ratio,
        "average_probability": avg_prob,
        "max_probability": max_prob,
        "min_probability": min_prob
    }

    result = {
        "is_deepfake": is_deepfake,
        "summary": summary,
        "top_frames": top_frames_base64,
        "metrics": metrics
    }
    
    # Removed the Tkinter display call so that only JSON is returned:
    # if app.config.get("DISPLAY_RESULTS", False):
    #     display_results(top_frames_base64, summary)

    return jsonify(result)

def display_results(top_frames_base64, summary):
    """
    Optional function to display results using Tkinter.
    """
    try:
        import tkinter as tk
        from PIL import ImageTk, Image
    except ImportError:
        print("Tkinter is not available.")
        return

    root = tk.Tk()
    root.title("Deepfake Detection Results")

    summary_label = tk.Label(root, text=summary, wraplength=600, justify="left")
    summary_label.pack(pady=10)

    for idx, img_b64 in enumerate(top_frames_base64):
        img_data = base64.b64decode(img_b64)
        pil_image = Image.open(BytesIO(img_data))
        tk_image = ImageTk.PhotoImage(pil_image)
        img_label = tk.Label(root, image=tk_image)
        img_label.image = tk_image  # Keep reference
        img_label.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Detection API")
    parser.add_argument("--display", action="store_true", help="Display results in a Tkinter window (for debugging only)")
    args = parser.parse_args()
    # For API use (frontend integration), ensure DISPLAY_RESULTS is False
    app.config["DISPLAY_RESULTS"] = args.display if args.display else False
    app.run(debug=True, port=5000)
