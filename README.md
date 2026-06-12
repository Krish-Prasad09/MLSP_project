# EmoSense — AI Interview Analyser

An AI-powered mock interview coach that analyses your speech — detecting emotional tone, communication quality, filler words, grammar, speaking pace, and vocabulary — and gives you a personalised interview score with coaching tips.

🔗 **Live Demo:** [https://huggingface.co/spaces/Krizzh/EmoSense-Interview-Analyser](https://huggingface.co/spaces/Krizzh/EmoSense-Interview-Analyser)

---

## Features

- 🎙️ Upload audio file or record live from microphone
- 😠 Detects 6 emotions: angry · disgust · fear · happy · neutral · sad
- 📊 Interview score out of 100 with emotion timeline
- 💬 Full speech transcription using Whisper
- ✅ Filler word detection (um, uh, like, basically...)
- 📝 Grammar issue detection
- 🏃 Speaking pace analysis (words per minute)
- 📖 Vocabulary diversity scoring
- 💡 Personalised coaching tips per session

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web Framework** | Flask (Python) | Backend server and REST API routes |
| **Emotion Recognition** | Wav2Vec2ForSequenceClassification | Fine-tuned transformer for speech emotion recognition |
| **Base Speech Model** | facebook/wav2vec2-base | Pre-trained speech representations from Facebook AI |
| **Speech-to-Text** | openai/whisper-small | Transcription of audio to text |
| **Audio Processing** | torchaudio | Audio loading, resampling, and feature extraction |
| **Audio Format Handling** | FFmpeg | Decoding browser-recorded WebM/audio formats |
| **Deep Learning** | PyTorch 2.2.2 | Model inference backend |
| **NLP (Grammar)** | LanguageTool (via language-tool-python) | Grammar error detection |
| **Frontend** | HTML + CSS + JavaScript | Single-page UI with mic recording and results display |
| **Training Dataset** | CREMA-D | 7,442 emotional speech clips from 91 actors |
| **Model Hosting** | Hugging Face Hub | Remote model storage and auto-download |
| **Deployment** | Hugging Face Spaces (Docker) | Cloud hosting with public URL |
| **Containerisation** | Docker | Reproducible deployment environment |

---

## Model Details

| Property | Value |
|---|---|
| Architecture | Wav2Vec2ForSequenceClassification |
| Base model | facebook/wav2vec2-base |
| Dataset | CREMA-D (7,442 clips · 91 actors) |
| Emotions | angry · disgust · fear · happy · neutral · sad |
| Split | 80% train · 10% val · 10% test |
| Accuracy | ~76% on test set |
| STT model | openai/whisper-small |
| Model hosted at | [Hugging Face Hub](https://huggingface.co/Krizzh/emosense-wav2vec2-crema-d) |

---

## Project Structure

```
Emosense/
├── EmoSense-Interview-Analyser/    ← Hugging Face Spaces deployment repo
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   ├── Dockerfile
│   ├── emotion.py
│   ├── requirements.txt
│   ├── speech_analysis.py
│   ├── .gitattributes
│   └── README                      ← HF Spaces config README
├── ser_wav2vec2_best/              ← trained model (not on GitHub, auto-downloaded from HF Hub)
├── templates/
│   └── index.html                  ← full frontend UI
├── app.py                          ← Flask server + API routes
├── emotion.py                      ← Wav2Vec2 inference + emotion timeline
├── speech_analysis.py              ← Whisper transcription + communication scoring
├── requirements.txt                ← all dependencies
├── SER_CREMA_D_Wav2Vec2.ipynb      ← Colab training notebook
├── .gitignore
└── README                          ← this file
```

> The `EmoSense-Interview-Analyser/` subfolder is the cloned Hugging Face Space repo used for deployment — it mirrors the core files needed to run the app in Docker.

---

## How to Run Locally

### Prerequisites

**Python 3.10 (Required)**

This project requires **exactly Python 3.10**. Newer versions (3.11, 3.12, 3.14) will cause `torch` installation errors.

Download Python 3.10 here:
👉 https://www.python.org/downloads/release/python-31011/

- **Windows** → download **Windows installer (64-bit)**
- **Mac** → download **macOS 64-bit universal2 installer**

> ⚠️ During installation on Windows — check ✅ **"Add Python to PATH"**

Verify after installing:
```powershell
py -3.10 --version
# should print: Python 3.10.x
```

**FFmpeg (Required for microphone recording)**

FFmpeg is needed to process audio recorded from the browser mic (WebM format).

- **Windows:** `winget install ffmpeg` — then restart PowerShell and verify: `ffmpeg -version`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

### Getting the Model

The trained Wav2Vec2 model is hosted on Hugging Face Hub and is **downloaded automatically on first run** — no manual steps needed.

If you prefer to train it yourself (~25 min on free Colab GPU):

1. Open `SER_CREMA_D_Wav2Vec2.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Go to **Runtime → Run all**
4. When prompted, mount your Google Drive and allow access
5. The model saves to `MyDrive/ser_wav2vec2_best/` on your Drive

Then download the folder, unzip it, and rename `processor_config.json` → `preprocessor_config.json` inside the folder. Place it in the project root as `ser_wav2vec2_best/` and update `MODEL_PATH` in `emotion.py` to `"./ser_wav2vec2_best"`.

The folder should contain exactly these 5 files:
```
ser_wav2vec2_best/
├── model.safetensors
├── config.json
├── preprocessor_config.json     ← renamed from processor_config.json
├── tokenizer_config.json
└── vocab.json
```

---

### Setup and Run

> ⚠️ Install torch separately first — this ensures the correct version is pulled from PyTorch's own servers.

**Windows**
```powershell
py -3.10 -m venv venv
venv\Scripts\activate
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

**Mac / Linux**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

> ⏳ First run will auto-download Whisper (~500 MB) and Wav2Vec2 base (~360 MB) — cached after first download.

Open **http://127.0.0.1:5000** in your browser.

> ⚠️ Keep the terminal window open while using the app — closing it stops the server.

---

## Deployment on Hugging Face Spaces

The app is deployed as a Docker Space on Hugging Face. The `EmoSense-Interview-Analyser/` folder is a cloned HF Space repo — pushing to it triggers an automatic rebuild and deploy.

The Dockerfile installs FFmpeg and PyTorch CPU, copies the app files, and starts Flask on port `7860` (required by HF Spaces). The Wav2Vec2 model is loaded directly from HF Hub on container startup — no model files are bundled in the repo.

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

---

## Common Errors

| Error | Fix |
|---|---|
| `OSError: preprocessor_config.json not found` | Rename `processor_config.json` → `preprocessor_config.json` inside `ser_wav2vec2_best/` |
| `py -3.10 is not recognized` | Python 3.10 not installed — download from link above |
| `ffmpeg not found` | Install ffmpeg — see Prerequisites section |
| `ModuleNotFoundError` | Make sure venv is activated before running |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |

---


