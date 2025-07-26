from flask import Flask, request, jsonify, send_file
import os
import re
from PIL import Image
import PyPDF2
from gtts import gTTS
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'uploads/'
AUDIO_FOLDER = 'audio/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Translation model
translator_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
translator_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ----- UTILS -----
def clean_text(text):
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\u2022+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(path):
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ''.join([page.extract_text() or '' for page in reader.pages])
    except:
        return ''

def translate_text(text, target_lang):
    try:
        lang_map = {"en": "en_XX", "ta": "ta_IN", "hi": "hi_IN"}
        forced_lang_id = translator_tokenizer.lang_code_to_id.get(lang_map.get(target_lang, "en_XX"), 250004)
        inputs = translator_tokenizer(text, return_tensors="pt")
        outputs = translator_model.generate(**inputs, forced_bos_token_id=forced_lang_id)
        return translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Translation error: {e}"

def text_to_audio(text, lang='ta'):
    try:
        audio_file = os.path.join(AUDIO_FOLDER, 'translated_audio.mp3')
        tts = gTTS(text=text, lang=lang)
        tts.save(audio_file)
        return audio_file
    except:
        return None

# ----- ROUTES -----

@app.route('/upload-file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    saved_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(saved_path)

    # Handle PDF and TXT
    if file_ext == ".pdf":
        text = extract_text_from_pdf(saved_path)
    elif file_ext == ".txt":
        with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    cleaned = clean_text(text)
    return jsonify({"text": cleaned})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text')
    target_lang = data.get('target_lang', 'en')

    if not text:
        return jsonify({"error": "Missing text"}), 400

    translated = translate_text(text, target_lang)
    return jsonify({"translated": translated})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        text = text[:1024]  # Limit input length
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]
        return jsonify({
            "brief_summary": summary["summary_text"],
            "detailed_summary": summary["summary_text"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path)
    return "Audio not found", 404

# ----- RUN -----
if __name__ == '__main__':
    app.run(debug=True, port=5000)
