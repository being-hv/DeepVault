# Deepfake Image Detection using GANs and SGANs

A full-stack, AI-powered system designed to detect deepfake images utilizing Generative Adversarial Networks (GANs) and Semi-Supervised GANs (SGANs) with a ResNet50 backbone.

## Architecture

- **Frontend**: Next.js 15, React, Tailwind CSS, TypeScript
- **Backend**: FastAPI, Python 3.10
- **Deep Learning**: PyTorch, Torchvision
- **Explainability**: Grad-CAM

## Features

- **SGAN Classifier**: Classifies images as Real or Fake, leveraging features learned by a ResNet50 backbone.
- **Grad-CAM Visualization**: Highlights the regions of the image that the model focused on to make its prediction.
- **Modern UI**: A sleek, dark-mode inspired frontend built with Next.js and Tailwind, offering drag-and-drop upload capabilities.

## Setup Instructions

### 1. Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Backend (FastAPI)
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 3. Frontend (Next.js)
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:3000`.

### 4. Docker (Backend)
To run the backend in a Docker container:
```bash
docker build -t deepfake-api .
docker run -p 8000:8000 deepfake-api
```

## Training the Model

1. Download a deepfake dataset (e.g., from Kaggle) and structure it as follows:
   ```
   datasets/data/
   ├── train/
   │   ├── real/
   │   └── fake/
   ├── val/
   │   ├── real/
   │   └── fake/
   ```
2. Run the training script:
   ```bash
   python -m training.train_sgan --data_dir ./datasets/data --epochs 50 --batch_size 32
   ```

## Deployment

- **Frontend**: Easily deployable to Vercel via GitHub integration.
- **Backend**: Can be deployed to Render or Railway using the provided `Dockerfile`.
# DeepVault
