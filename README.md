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
###Step 2: Set Up the Backend
-- **Go to the backend folder:**
```bash
cd backend
```
-- **Create and activate a virtual environment:**
--- Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
--- macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
-- **Install Python packages:**
```bash
pip install -r requirements.txt
```
### Step 3: Set Up the Frontend
-- **Go to the frontend folder:**
```bash
cd ../frontend
```
-- **Install npm packages:**
```bash
npm install
```
### Step 4: Add Pre-trained Models
-- **Download the StyleGAN2 model weights from NVIDIA's repository and place them in backend/models.**

## Usage
### Start the Backend
-- **In the backend folder, with the virtual environment active, run:**
```bash
python app.py
```
-- Backend runs at http://localhost:5000.
-- Start the Frontend
-- **In the frontend folder, run:**
```bash
npm start
```
-- Frontend runs at http://localhost:3000.
-- **Use the App**
-- 1. Open http://localhost:3000 in your browser.
-- 2. Upload a video.
-- 3. See the results, including any frames flagged as fake.

## Configuration
### Backend
-- API URL: Frontend talks to http://localhost:5000. Change this in the frontend if needed.
-- Model Path: Check that app.py points to the right model file.
###Frontend
-- Environment Variables: Add a .env file in frontend for settings like the API URL.

## Contributing
### Want to help? Here’s how:
-- 1. Fork this repository.
-- 2. Create a branch for your changes.
-- 3. Submit a pull request with details of what you did.

##License
###This project uses the MIT License.

###Acknowledgements
-- StyleGAN2 by NVIDIA
-- Flask
-- React
