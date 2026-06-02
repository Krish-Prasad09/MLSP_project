from flask import Flask, request, render_template, jsonify
from emotion import predict_emotion

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm", "m4a", "flac"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "" and audio_file.content_length == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        result = predict_emotion(audio_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse_speech", methods=["POST"])
def analyse_speech_route():
    """
    Speech-to-text + communication quality analysis.
    Accepts same audio formats as /predict.
    Returns: transcript, highlighted_transcript, scores, tips, etc.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        from speech_analysis import analyse_speech
        result = analyse_speech(audio_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyse_full", methods=["POST"])
def analyse_full():
    """
    Combined endpoint: runs emotion analysis AND speech analysis on the
    same audio in one request. Returns both results together so the
    frontend can display a unified interview report.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    try:
        audio_bytes = audio_file.read()
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio data"}), 400

        # ── Emotion analysis (Wav2Vec2) ────────────────────────────────────
        emotion_result = predict_emotion(audio_bytes)

        # ── Speech analysis (Whisper + rule engine) ────────────────────────
        from speech_analysis import analyse_speech
        speech_result = analyse_speech(audio_bytes)

        return jsonify({
            "emotion": emotion_result,
            "speech":  speech_result,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(" EmoSense by Group 17 starting at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
    # app.run(debug=False, host="0.0.0.0", port=7860)

    