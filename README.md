# Deepfake Detection using StyleGAN2

**A web application for detecting deepfake videos using machine learning techniques.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview
This project is a web-based tool that detects deepfake videos. It uses a StyleGAN2 discriminator model to analyze video frames and determine if they are real or fake. The app has a React frontend for users to interact with and a Flask backend to process videos.

---

## Features
- **Video Upload:** Upload videos through the web interface.
- **Frame Extraction:** Extracts frames from videos for analysis.
- **Deepfake Detection:** Classifies frames as real or fake using StyleGAN2.
- **Results:** Displays if the video is a deepfake and shows suspect frames.
- **Hardware Friendly:** Works well on mid-range GPUs like the NVIDIA 1660Ti Max-Q.

---

## Technologies Used
- **Frontend:** React.js
- **Backend:** Flask (Python)
- **Machine Learning:** PyTorch, StyleGAN2
- **Image Processing:** OpenCV, Pillow
- **API:** Axios

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Node.js and npm
- Git (optional, for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Adhithya070/MiniProj.git
cd MiniProj
```

### Step 2: Clone NVIDIA’s StyleGAN2-ADA-PyTorch Repository
```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git backend/stylegan2-ada-pytorch
```
### Step 3: Set Up the Backend
1. **Go to the backend folder:**
```bash
cd backend
```
2. **Create and activate a virtual environment:**
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. **Install Python packages:**
```bash
pip install -r requirements.txt
```

### Step 4: Set Up the Frontend
1. **Go to the frontend folder:**
```bash
cd ../frontend
```
2. **Install npm packages:**
```bash
npm install
```

### Step 4: Add Pre-trained Models
1. **Download the StyleGAN2-ADA model weights (e.g., pretrained.pkl) from [NVIDIA’s official repository](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/).**
2. **Place the downloaded model file in the backend/models directory.**

## Usage
### **Start the Backend**
1. **In the backend folder, activate the virtual environment if not already active:**
- **Windows:**
```bash
venv\Scripts\activate
```
- **macOS/Linux:**
```bash
source venv/bin/activate
```
2. Run the Flask server:
```bash
python app.py
```
- The backend will run at `http://localhost:5000`.

### **Start the Frontend**
- **In the frontend folder, run:**
```bash
npm start
```
- Frontend runs at `http://localhost:3000`.
  
### **Use the App**
1. Open `http://localhost:3000` in your browser.
2. Upload a video file (MP4, AVI, or MOV).
3. Wait for the backend to process the video and display the results.
4. Review the detection results, including flagged frames.
---

## DEMO
![recdemo](https://github.com/user-attachments/assets/0a5e77ad-6296-4ddd-af01-21fe86fc0593)

## Configuration
### Backend
- API Endpoint: The frontend communicates with `http://localhost:5000` by default. Update this in `frontend/src/api.js` if needed.
- Model Path: Ensure app.py points to the correct model file in `backend/models`.
###Frontend
- Environment Variables: Create a `.env` file in the frontend folder to customize settings (e.g., REACT_APP_API_URL=`http://localhost:5000`).

## Contributing
### Want to help? Here’s how:
1. Fork this repository.
2. Create a branch for your changes.
3. Submit a pull request with details of what you did.

## License
### This project uses the MIT License. See [LICENSE]((https://choosealicense.com/licenses/mit/)) for details.

### Acknowledgements
- [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) by NVIDIA
- Flask
- React
