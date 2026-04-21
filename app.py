from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pytesseract

from ocr_module import extract_text
from normalization_module import SPELLCHECK_AVAILABLE
from hybride_feature import extract_hybrid_features
from llm_module import postprocess_text, evaluate_with_ollama
from scorefeedback_module import parse_evaluation_result

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- TESSERACT ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- FILE CHECK ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    evaluation = ""
    question = ""
    reference = ""
    processed_image = None
    score = ""
    feedback = ""
    features = {}

    if request.method == "POST":
        question = request.form.get("question", "")
        reference = request.form.get("reference", "")
        extracted_text = request.form.get("extracted_text", "")
        processed_image = request.form.get("processed_image")

        # -------- EXTRACT --------
        if "extract" in request.form:
            file = request.files.get("file")

            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                raw_text, processed_path = extract_text(file_path, app.config["UPLOAD_FOLDER"])
                extracted_text = postprocess_text(raw_text)

                if processed_path:
                    processed_image = os.path.basename(processed_path)
            else:
                extracted_text = "No valid file selected."

        # -------- EVALUATE --------
        elif "evaluate" in request.form:
            if extracted_text.strip() and question.strip() and reference.strip():
                features = extract_hybrid_features(reference, extracted_text)
                evaluation = evaluate_with_ollama(extracted_text, reference, question)
                score, feedback = parse_evaluation_result(evaluation)
            else:
                feedback = "Missing extracted text, question, or reference answer."

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        evaluation=evaluation,
        question=question,
        reference=reference,
        processed_image=processed_image,
        spellcheck_enabled=SPELLCHECK_AVAILABLE,
        score=score,
        feedback=feedback,
        features=features
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)