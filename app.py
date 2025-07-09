
import os
from flask import Flask, request, render_template, send_file
from moviepy.editor import VideoFileClip
import whisper
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from fpdf import FPDF

import nltk
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
TRANSCRIPT_FOLDER = "transcripts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

model = whisper.load_model("base")

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def extract_keywords(text, top_n=10):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w.isalpha() and w not in stop_words]
    most_common = Counter(filtered).most_common(top_n)
    return [word for word, freq in most_common]

def summarize(text, num_sentences=5):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:num_sentences])

def save_as_txt(text, filename):
    path = os.path.join(TRANSCRIPT_FOLDER, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    path = os.path.join(TRANSCRIPT_FOLDER, filename)
    pdf.output(path)
    return path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["video"]
        if not file:
            return "Файл не загружен", 400

        filename = file.filename
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_path = video_path.replace(".mp4", ".wav")
        file.save(video_path)

        extract_audio(video_path, audio_path)
        full_text = transcribe(audio_path)
        keywords = extract_keywords(full_text)
        summary = summarize(full_text)

        txt_path = save_as_txt(full_text, filename + ".txt")
        pdf_path = save_as_pdf(full_text, filename + ".pdf")

        return render_template("index.html", transcript=full_text, summary=summary,
                               keywords=keywords, txt_file=txt_path, pdf_file=pdf_path)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(TRANSCRIPT_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
