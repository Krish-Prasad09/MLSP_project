# EmoSense — AI Interview Analyser

An AI-powered mock interview coach that analyses your speech in real time —
detecting emotional tone, communication quality, filler words, grammar, pace,
and vocabulary — and gives you a personalised interview score with coaching tips.

**Built with:** Wav2Vec2 (fine-tuned on CREMA-D) · Whisper STT · Flask

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

## Prerequisites

### 1. Python 3.10 (Required)

This project requires **exactly Python 3.10**.  
Newer versions (3.11, 3.12, 3.14) will cause `torch` installation errors.

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

### 2. FFmpeg (Required for microphone recording)

FFmpeg is needed to process audio recorded from the browser mic (WebM format).

- **Windows:** `winget install ffmpeg`  
  Then restart PowerShell and verify: `ffmpeg -version`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

## Getting the Model

The trained Wav2Vec2 model is not included in this repo (too large for GitHub).  
You need to train it yourself using the provided Colab notebook — takes ~25 min on free Colab GPU.

### Step 1 — Train the model on Google Colab

1. Open `SER_CREMA_D_Wav2Vec2.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Go to **Runtime → Run all**
4. When prompted, mount your Google Drive and allow access
5. Wait for training to complete (~25 minutes)

The model will be saved automatically to your Google Drive at:
```
MyDrive/ser_wav2vec2_bestnew/
```

### Step 2 — Download from Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Find the `ser_wav2vec2_bestnew` folder
3. Right-click → **Download** (downloads as a zip file)
4. Unzip it

### Step 3 — Rename files correctly

1. Rename the unzipped **folder** from `ser_wav2vec2_bestnew` → `ser_wav2vec2_best`
2. Open the folder and rename `processor_config.json` → `preprocessor_config.json`

> ⚠️ Both renames are required — the app will crash without them.

The folder should contain exactly these 5 files:
```
ser_wav2vec2_best/
├── model.safetensors
├── config.json
├── preprocessor_config.json     ← renamed from processor_config.json
├── tokenizer_config.json
└── vocab.json
```

### Step 4 — Place the folder in the project

```
emosense/
├── ser_wav2vec2_best/           ← paste here
├── templates/
├── app.py
├── emotion.py
├── speech_analysis.py
├── requirements.txt
├── run.bat
└── SER_CREMA_D_Wav2Vec2.ipynb
```

---

## Setup

### Windows

```powershell
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Mac / Linux

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

> ⏳ First run will auto-download:
> - Whisper STT model (~500 MB)  
> - Wav2Vec2 base model (~360 MB)  
> These are cached after the first download.

---

## Running the App

Open **http://127.0.0.1:5000** in Chrome.

> ⚠️ Keep the terminal window open while using the app — closing it stops the server.

---

## Project Structure

```
emosense/
├── ser_wav2vec2_best/          ← your trained model (not on GitHub)
├── templates/
│   └── index.html              ← full frontend UI
├── app.py                      ← Flask server + API routes
├── emotion.py                  ← Wav2Vec2 inference + emotion timeline
├── speech_analysis.py          ← Whisper transcription + communication scoring
├── requirements.txt            ← all dependencies
├── run.bat                     ← one-click launcher (Windows)
├── .gitignore
└── SER_CREMA_D_Wav2Vec2.ipynb  ← Colab training notebook
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | Wav2Vec2ForSequenceClassification |
| Base model | facebook/wav2vec2-base |
| Dataset | CREMA-D (7,442 clips · 91 actors) |
| Emotions | angry · disgust · fear · happy · neutral · sad |
| Split | 70% train · 15% val · 15% test |
| Accuracy | ~76% on test set |
| STT model | openai/whisper-small |

---

## Common Errors

| Error | Fix |
|---|---|
| `OSError: preprocessor_config.json not found` | Rename `processor_config.json` → `preprocessor_config.json` inside `ser_wav2vec2_best/` |
| `py -3.10 is not recognized` | Python 3.10 not installed — download from link above |
| `ffmpeg not found` | Install ffmpeg — see Prerequisites section |
| `ModuleNotFoundError` | Make sure venv is activated before running |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |
| Mic not working | Your personal issue |

---

## Tech Stack

- **Flask** — lightweight Python web framework
- **Wav2Vec2** — Facebook's transformer model for speech, fine-tuned for emotion
- **Whisper** — OpenAI's speech-to-text model (runs fully locally)
- **torchaudio** — audio loading and resampling
- **CREMA-D** — Crowd Sourced Emotional Multimodal Actors Dataset

---

*Made by Group 17*
