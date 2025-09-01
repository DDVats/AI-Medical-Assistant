# 🩺 Advanced AI Medical Assistant

A multi-functional AI-powered medical assistant for diagnostic support of cardiovascular and pulmonary conditions. It analyzes ECG signals and chest X-ray images via a user-friendly Streamlit interface.

## Features
- Dual Diagnostic Modules: Switch between Heart (ECG) and Lung (X-ray) AI models.
- ❤️ Heart Disease Detection: Predicts Congestive Heart Failure (CHF) from ECG `.dat` and `.hea` files using a TensorFlow/Keras model.
- 🫁 Lung Disease Detection: Predicts Pneumonia from chest X-ray images using a Vision Transformer from Hugging Face.
- Interactive UI: Streamlit front-end for file uploads and result visualization.
- API-Driven Architecture: Flask back-end serving AI models.

## Prerequisites
- Git: https://git-scm.com/downloads  
- Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/  
- Git LFS: https://git-lfs.com

## Setup Instructions

### 1. Clone the Repository
```bash
git lfs clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
