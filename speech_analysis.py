"""
speech_analysis.py
──────────────────
Speech-to-text (Whisper) + Communication Quality Analysis for EmoSense.

Scores reported:
  • Filler words     — "um", "uh", "like", "you know", "basically", etc.
  • Grammar          — rule-based checks (subject-verb agreement, double negatives, etc.)
  • Sentence clarity — avg words-per-sentence, passive-voice ratio, long-sentence ratio
  • Vocabulary       — type-token ratio (lexical diversity)
  • Pace             — words per ACTIVE SPEAKING MINUTE (not total clip length).
                       Whisper timestamps are used to measure only the segments where
                       the candidate is actually speaking, so a 5-minute recording where
                       the interviewer talks for 3 minutes still gives a fair WPM reading.
  • Overall          — weighted composite

All of this runs 100 % locally (no API key needed).
Whisper model is downloaded once and cached by HuggingFace.
"""

import re
import math
import os
import io
import tempfile
import subprocess
import numpy as np

# ── Lazy-load Whisper so app starts fast ──────────────────────────────────
# _whisper_pipe = None

# def _get_whisper():
#     global _whisper_pipe
#     if _whisper_pipe is None:
#         from transformers import pipeline
#         print("Loading Whisper STT model (first run — downloading ~150 MB)…")
#         _whisper_pipe = pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-small",          # ~150 MB, good accuracy/speed
#             generate_kwargs={"language": "english"},
#             return_timestamps=True,
#         )
#         print("✅ Whisper loaded.")
#     return _whisper_pipe


# ── Audio helpers ──────────────────────────────────────────────────────────

# ── Load Whisper at startup (same as emotion model) ───────────────────────
from transformers import pipeline
print("Loading Whisper STT model...")
_whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"language": "english"},
    return_timestamps=True,
)
print("✅ Whisper loaded.")

def _get_whisper():
    return _whisper_pipe

def _detect_suffix(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"RIFF":               return ".wav"
    if audio_bytes[:3] == b"ID3":                return ".mp3"
    if audio_bytes[:4] == b"fLaC":               return ".flac"
    if audio_bytes[:4] == b"OggS":               return ".ogg"
    if audio_bytes[4:8] == b"ftyp":              return ".m4a"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":  return ".webm"
    return ".webm"


def _to_wav_numpy(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert any audio format → numpy float32 array at 16 kHz mono.
    Returns (array, sample_rate).
    """
    import torchaudio, torch

    # Try torchaudio first
    try:
        wf, sr = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception:
        # Fall back to ffmpeg conversion
        suffix  = _detect_suffix(audio_bytes)
        tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav",  delete=False)
        tmp_in.close(); tmp_out.close()
        try:
            with open(tmp_in.name, "wb") as f: f.write(audio_bytes)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in.name,
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_out.name],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg: {result.stderr.decode()}")
            wf, sr = torchaudio.load(tmp_out.name)
        finally:
            for p in (tmp_in.name, tmp_out.name):
                if os.path.exists(p): os.remove(p)

    if sr != 16000:
        wf = torchaudio.functional.resample(wf, sr, 16000)
        sr = 16000
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    return wf.squeeze().numpy().astype(np.float32), sr


# ── Transcription ──────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> dict:
    """
    Run Whisper on audio_bytes.

    Returns:
        text               – full transcript string
        chunks             – raw Whisper chunk list (each has 'timestamp')
        duration_seconds   – total length of the recording
        speaking_seconds   – sum of durations of speech segments only.
                             This is the denominator used for WPM so that
                             silence / interviewer questions don't make the
                             candidate look like they were speaking slowly.
    """
    audio_np, sr = _to_wav_numpy(audio_bytes)
    duration_sec = len(audio_np) / sr

    pipe = _get_whisper()
    result = pipe(
        {"array": audio_np, "sampling_rate": sr},
        chunk_length_s=30,
        stride_length_s=5,
    )

    text   = result.get("text", "").strip()
    chunks = result.get("chunks", [])

    # ── Measure actual speaking time from per-chunk timestamps ────────────
    # Whisper returns chunks like: {"timestamp": [0.0, 2.4], "text": "..."}
    # Summing those durations gives us the time the candidate was actually
    # speaking — not including pauses, silence, or the interviewer's turns.
    speaking_seconds = 0.0
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if ts and len(ts) == 2 and ts[0] is not None and ts[1] is not None:
            chunk_dur = float(ts[1]) - float(ts[0])
            if chunk_dur > 0:
                speaking_seconds += chunk_dur

    # Fallback: if no timestamps came back, use total duration
    if speaking_seconds < 1.0:
        speaking_seconds = duration_sec

    return {
        "text":             text,
        "chunks":           chunks,
        "duration_seconds": round(duration_sec, 1),
        "speaking_seconds": round(speaking_seconds, 1),
    }


# ── Communication analysis ─────────────────────────────────────────────────

# Filler words — expanded list
FILLERS = {
    "um", "uh", "er", "ah", "hmm", "umm", "uhh", "err",
    "like", "basically", "literally", "honestly", "actually",
    "you know", "you know what i mean", "i mean",
    "sort of", "kind of", "kinda", "sorta",
    "right", "okay so", "so yeah", "and stuff",
    "at the end of the day", "to be honest", "to be fair",
    "well", "anyway", "whatever",
}

# Multi-word fillers (checked before single-word)
MULTI_FILLERS = sorted(
    [f for f in FILLERS if " " in f],
    key=lambda x: -len(x)   # longest first
)

SINGLE_FILLERS = {f for f in FILLERS if " " not in f}

# Weak/vague words
WEAK_WORDS = {
    "thing", "stuff", "things", "lot", "lots", "very", "really",
    "quite", "pretty", "just", "maybe", "perhaps", "probably",
    "might", "could", "somewhat", "a bit",
}

# Grammar patterns (regex → description)
GRAMMAR_ISSUES = [
    (r"\bi done\b",            "Incorrect: 'I done' → use 'I did' or 'I have done'"),
    (r"\bthey was\b",          "Subject-verb disagreement: 'they was' → 'they were'"),
    (r"\bwe was\b",            "Subject-verb disagreement: 'we was' → 'we were'"),
    (r"\byou was\b",           "Subject-verb disagreement: 'you was' → 'you were'"),
    (r"\bi seen\b",            "Incorrect: 'I seen' → 'I saw' or 'I have seen'"),
    (r"\bi have went\b",       "Incorrect: 'I have went' → 'I have gone'"),
    (r"\bshould of\b",         "Common error: 'should of' → 'should have'"),
    (r"\bwould of\b",          "Common error: 'would of' → 'would have'"),
    (r"\bcould of\b",          "Common error: 'could of' → 'could have'"),
    (r"\bmore better\b",       "Double comparative: 'more better' → 'better'"),
    (r"\bmost best\b",         "Double superlative: 'most best' → 'best'"),
    (r"\bdon't have no\b",     "Double negative detected"),
    (r"\bcan't do nothing\b",  "Double negative detected"),
    (r"\bain't\b",             "Informal: 'ain't' — avoid in interviews"),
    (r"\bgonna\b",             "Informal: 'gonna' → 'going to'"),
    (r"\bwanna\b",             "Informal: 'wanna' → 'want to'"),
    (r"\bgotta\b",             "Informal: 'gotta' → 'have to' / 'got to'"),
    (r"\bdunno\b",             "Informal: 'dunno' → 'I don't know'"),
    (r"\byeah\b",              "Informal: 'yeah' → 'yes'"),
    (r"\bnope\b",              "Informal: 'nope' → 'no'"),
]

PASSIVE_PATTERN = re.compile(
    r"\b(was|were|is|are|been|be|being)\s+(being\s+)?\w+ed\b", re.IGNORECASE
)


def _find_fillers(text_lower: str) -> list[str]:
    found = []
    temp  = text_lower

    # Multi-word first
    for mf in MULTI_FILLERS:
        count = temp.count(mf)
        for _ in range(count):
            found.append(mf)
        temp = temp.replace(mf, " " * len(mf))

    # Single-word
    words = re.findall(r"\b\w+\b", temp)
    for w in words:
        if w in SINGLE_FILLERS:
            found.append(w)

    return found


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def analyse_communication(transcript_text: str, speaking_seconds: float) -> dict:
    """
    Full communication quality analysis.

    speaking_seconds should be the ACTIVE SPEAKING TIME derived from
    Whisper timestamps — not the total clip duration. This ensures WPM
    reflects the candidate's delivery speed only, unaffected by the
    interviewer asking questions or natural pauses between turns.

    Returns a rich dict with scores and detailed feedback.
    """
    if not transcript_text or len(transcript_text.strip()) < 10:
        return {"error": "Transcript too short to analyse."}

    text       = transcript_text.strip()
    text_lower = text.lower()
    words      = re.findall(r"\b\w+\b", text_lower)
    sentences  = _sentences(text)

    word_count     = len(words)
    sentence_count = max(len(sentences), 1)

    if word_count < 5:
        return {"error": "Too few words to analyse (< 5)."}

    # ── 1. Filler words ───────────────────────────────────────────────────
    filler_instances = _find_fillers(text_lower)
    filler_count     = len(filler_instances)
    filler_rate      = filler_count / word_count  # fraction of words

    # Score: 0 fillers = 100, 20 %+ = 0
    filler_score = max(0, round(100 - (filler_rate / 0.20) * 100))
    filler_freq  = {}
    for f in filler_instances:
        filler_freq[f] = filler_freq.get(f, 0) + 1
    top_fillers = sorted(filler_freq.items(), key=lambda x: -x[1])[:5]

    # ── 2. Grammar ────────────────────────────────────────────────────────
    grammar_issues = []
    for pattern, msg in GRAMMAR_ISSUES:
        if re.search(pattern, text_lower):
            grammar_issues.append(msg)

    # Score: 0 issues = 100, each issue deducts 10 pts, min 0
    grammar_score = max(0, 100 - len(grammar_issues) * 10)

    # ── 3. Sentence clarity ───────────────────────────────────────────────
    avg_words_per_sent = word_count / sentence_count
    sentence_lengths   = [len(re.findall(r"\b\w+\b", s)) for s in sentences]

    # Long sentences (> 35 words) are hard to follow in speech
    long_sent_count = sum(1 for l in sentence_lengths if l > 35)
    long_sent_ratio = long_sent_count / sentence_count

    # Passive voice ratio
    passive_matches = PASSIVE_PATTERN.findall(text)
    passive_ratio   = len(passive_matches) / sentence_count

    # Weak/vague word ratio
    weak_count = sum(1 for w in words if w in WEAK_WORDS)
    weak_ratio = weak_count / word_count

    # Clarity score — penalise long sentences, passive voice, weak words
    clarity_score = 100
    clarity_score -= long_sent_ratio  * 30
    clarity_score -= passive_ratio    * 20
    clarity_score -= weak_ratio       * 30
    # Ideal avg sentence length 12–20 words
    if avg_words_per_sent > 25:
        clarity_score -= (avg_words_per_sent - 25) * 1.5
    elif avg_words_per_sent < 6:
        clarity_score -= (6 - avg_words_per_sent) * 3
    clarity_score = max(0, min(100, round(clarity_score)))

    # ── 4. Vocabulary diversity (type-token ratio) ────────────────────────
    unique_words  = set(words)
    ttr           = len(unique_words) / word_count   # 0–1

    # Corrected TTR (normalise over 50-word window to be fair for long texts)
    window = 50
    if word_count >= window:
        cttr = len(set(words[:window])) / window
    else:
        cttr = ttr

    vocab_score = min(100, round(cttr * 150))   # 67 % unique → 100

    # ── 5. Pace — based on active speaking time only ──────────────────────
    # We use speaking_seconds (sum of Whisper chunk timestamps) rather than
    # total recording duration. This is interview-accurate: a candidate who
    # answers a 2-minute question in a 10-minute session should be judged on
    # how fast they spoke during those 2 minutes, not the whole session.
    wpm = (word_count / speaking_seconds) * 60 if speaking_seconds > 0 else 0

    # Ideal interview pace: 130–160 wpm
    IDEAL_LOW, IDEAL_HIGH = 130, 160
    if IDEAL_LOW <= wpm <= IDEAL_HIGH:
        pace_score = 100
    elif wpm < IDEAL_LOW:
        pace_score = max(0, round(100 - (IDEAL_LOW - wpm) * 1.2))
    else:
        pace_score = max(0, round(100 - (wpm - IDEAL_HIGH) * 1.5))

    if wpm < 80:
        pace_feedback = "Very slow — try to be more fluent and natural."
    elif wpm < 130:
        pace_feedback = "Slightly slow — aim for a more conversational pace."
    elif wpm <= 160:
        pace_feedback = "Great pace — clear and easy to follow."
    elif wpm <= 200:
        pace_feedback = "A bit fast — slow down so the interviewer can follow."
    else:
        pace_feedback = "Too fast — significantly slow down your delivery."

    # ── 6. Composite score ────────────────────────────────────────────────
    overall_score = round(
        filler_score  * 0.25 +
        grammar_score * 0.25 +
        clarity_score * 0.25 +
        vocab_score   * 0.15 +
        pace_score    * 0.10
    )

    # ── 7. Summary tips ───────────────────────────────────────────────────
    tips = []
    if filler_score < 70:
        worst = top_fillers[0][0] if top_fillers else "filler words"
        tips.append(f"Reduce filler words — you used '{worst}' {top_fillers[0][1] if top_fillers else ''} times. Pause silently instead.")
    if grammar_score < 80 and grammar_issues:
        tips.append(f"Grammar: {grammar_issues[0]}")
    if clarity_score < 70:
        if long_sent_ratio > 0.2:
            tips.append("Break long sentences into shorter ones — aim for under 20 words per sentence.")
        if passive_ratio > 0.3:
            tips.append("Use active voice more: say 'I led the team' not 'The team was led by me'.")
    if vocab_score < 60:
        tips.append("Vary your vocabulary — avoid repeating the same words. Use specific, concrete language.")
    if pace_score < 70:
        tips.append(pace_feedback)

    if not tips:
        tips.append("Excellent communication! Keep up the confident, clear delivery.")

    # ── annotate transcript with filler highlights ─────────────────────────
    highlighted = text
    all_filler_words = sorted(set(filler_instances), key=lambda x: -len(x))
    for f in all_filler_words:
        pattern = re.compile(r'\b' + re.escape(f) + r'\b', re.IGNORECASE)
        highlighted = pattern.sub(f'<mark class="filler">{f}</mark>', highlighted)

    return {
        # Scores
        "overall_score":   overall_score,
        "filler_score":    filler_score,
        "grammar_score":   grammar_score,
        "clarity_score":   clarity_score,
        "vocab_score":     vocab_score,
        "pace_score":      pace_score,

        # Details
        "word_count":          word_count,
        "sentence_count":      sentence_count,
        "avg_words_per_sent":  round(avg_words_per_sent, 1),
        "filler_count":        filler_count,
        "filler_rate_pct":     round(filler_rate * 100, 1),
        "top_fillers":         [{"word": w, "count": c} for w, c in top_fillers],
        "grammar_issues":      grammar_issues,
        "passive_ratio_pct":   round(passive_ratio * 100, 1),
        "wpm":                 round(wpm, 1),
        "speaking_seconds":    round(speaking_seconds, 1),
        "pace_feedback":       pace_feedback,
        "tips":                tips,

        # Highlighted transcript
        "highlighted_transcript": highlighted,
    }


# ── Combined entry point ───────────────────────────────────────────────────

def analyse_speech(audio_bytes: bytes) -> dict:
    """
    Transcribe audio and run full communication analysis.
    Returns combined dict safe to jsonify().
    """
    stt            = transcribe(audio_bytes)
    text           = stt["text"]
    duration_sec   = stt["duration_seconds"]
    speaking_sec   = stt["speaking_seconds"]

    # Pass active speaking time so WPM is calculated correctly
    comm = analyse_communication(text, speaking_sec)

    return {
        "transcript":       text,
        "duration_seconds": duration_sec,
        "speaking_seconds": speaking_sec,
        "communication":    comm,
    }
