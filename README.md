# Deepfake Detection Project

**A web application for detecting deepfake videos using machine learning techniques.**

![Project Logo](https://via.placeholder.com/150) <!-- Replace with your project logo -->

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
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
Step 2: Clone NVIDIA’s StyleGAN2-ADA-PyTorch Repository
bash
Copy
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git backend/stylegan2-ada-pytorch
Step 3: Set Up the Backend
Navigate to the backend folder:

bash
Copy
cd backend
Create and activate a virtual environment:

Windows:

bash
Copy
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
Copy
python3 -m venv venv
source venv/bin/activate
Install Python packages:

bash
Copy
pip install -r requirements.txt
Step 4: Add Pre-trained Models
Download the StyleGAN2-ADA model weights (e.g., pretrained.pkl) from NVIDIA’s official repository.

Place the downloaded model file in the backend/models directory.

Step 5: Set Up the Frontend
Navigate to the frontend folder:

bash
Copy
cd ../frontend
Install npm packages:

bash
Copy
npm install
Usage
Start the Backend
In the backend folder, activate the virtual environment if not already active:

Windows:

bash
Copy
venv\Scripts\activate
macOS/Linux:

bash
Copy
source venv/bin/activate
Run the Flask server:

bash
Copy
python app.py
The backend will run at http://localhost:5000.

Start the Frontend
In the frontend folder, start the React app:

bash
Copy
npm start
The frontend will run at http://localhost:3000.

Use the Application
Open http://localhost:3000 in your browser.

Upload a video file (MP4, AVI, or MOV).

Wait for the backend to process the video and display the results.

Review the detection results, including flagged frames.

Configuration
Backend
API Endpoint: The frontend communicates with http://localhost:5000 by default. Update this in frontend/src/api.js if needed.

Model Path: Ensure app.py points to the correct model file in backend/models.

Frontend
Environment Variables: Create a .env file in the frontend folder to customize settings (e.g., REACT_APP_API_URL=http://localhost:5000).

Contributing
Fork this repository.

Create a feature branch:

bash
Copy
git checkout -b feature/your-feature
Commit your changes:

bash
Copy
git commit -m "Add your message"
Push to the branch:

bash
Copy
git push origin feature/your-feature
Open a pull request on GitHub.

License
This project is licensed under the MIT License. See LICENSE for details.

Acknowledgements
StyleGAN2-ADA-PyTorch by NVIDIA

Flask and React communities for documentation and tools.

