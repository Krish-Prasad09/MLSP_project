## Setup

>Python 3.10 is required. Python 3.11/3.12/3.14 may cause torch compatibility issues.

### Windows
```powershell
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python.py
```

### Mac/Linux
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python.py
```

> First run will auto-download Whisper (~500MB) and Wav2Vec2 base model.
> Your trained model must be placed in `ser_wav2vec2_best/` folder.
