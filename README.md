
```markdown
# Sinhala Meme Hate Speech Detection - Backend ⚙️

This repository contains the **FastAPI** backend and the **Multimodal Deep Learning Architecture** (ResNet50 + mBERT) for the Sinhala Meme Hate Speech Detection System. 

🔗 **Frontend Repository:** [මෙතනට ඔයාගේ Frontend Repo එකේ ලින්ක් එක දාන්න]

## 🛠️ Technology Stack
* FastAPI (Python)
* PyTorch & Hugging Face Transformers
* Tesseract OCR
* SQLite Database

## ⚙️ Prerequisites
* Python 3.9+
* Tesseract OCR installed locally (with the `sin.traineddata` Sinhala language pack).

## 🚀 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MAlindu-pabasar/sinhala-meme-backend.git](https://github.com/MAlindu-pabasar/sinhala-meme-backend.git)
   cd sinhala-meme-backend

2.**Set up the virtual environment:**
```bash
  python -m venv venv
  # Windows:
  venv\Scripts\activate
  # macOS/Linux:
  source venv/bin/activate

3.**Install dependencies:**
```bash
  pip install -r requirements.txt

4.**Configure Tesseract:**
  Ensure the OCR path in main.py matches your local Tesseract installation directory.

5.**Start the API Server:**

```bash
  uvicorn main:app --reload

  The API will be accessible at http://localhost:8000
