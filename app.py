from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import re
import requests
import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pdf2image import convert_from_path

# ---------------- OPTIONAL SPELL CHECKER ----------------
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    SPELLCHECK_AVAILABLE = True
except Exception:
    SPELLCHECK_AVAILABLE = False
    spell = None

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- TESSERACT ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- OLLAMA ----------------
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral:latest"

# ---------------- FILE CHECK ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- IMAGE HELPERS ----------------
def resize_for_ocr(pil_img, scale=1.5):
    w, h = pil_img.size
    return pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

def cv_to_pil(cv_img):
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# ---------------- PREPROCESSING ----------------
def generate_preprocessed_variants(image_path):
    """
    Faster preprocessing with 2 useful variants.
    """
    original = Image.open(image_path).convert("RGB")
    enlarged = resize_for_ocr(original, scale=1.5)

    gray = enlarged.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    variants = []
    variants.append(("gray", gray))

    gray_cv = np.array(gray)
    adaptive = cv2.adaptiveThreshold(
        gray_cv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12
    )
    variants.append(("adaptive", cv_to_pil(adaptive)))

    return variants

# ---------------- OCR HELPERS ----------------
def clean_ocr_noise(text):
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def score_ocr_result(text, confidences):
    text = clean_ocr_noise(text)
    words = re.findall(r"[A-Za-z]{2,}", text)

    valid_conf = [c for c in confidences if c != -1]
    avg_conf = sum(valid_conf) / len(valid_conf) if valid_conf else 0

    score = avg_conf
    score += min(len(words) * 0.8, 35)
    score += min(len(text) * 0.05, 20)

    return score

def run_tesseract_best(pil_img):
    """
    Faster OCR using 2 PSM modes.
    """
    best_text = ""
    best_score = -1
    psm_modes = [6, 11]

    for psm in psm_modes:
        config = f"--oem 3 --psm {psm} -l eng"
        try:
            data = pytesseract.image_to_data(
                pil_img,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            text = " ".join([t for t in data["text"] if t.strip()])

            confidences = []
            for conf in data["conf"]:
                try:
                    confidences.append(float(conf))
                except Exception:
                    pass

            current_score = score_ocr_result(text, confidences)

            if current_score > best_score:
                best_score = current_score
                best_text = text

        except Exception:
            continue

    return clean_ocr_noise(best_text), best_score

def save_best_processed_variant(variants, save_base_path):
    preferred_order = ["adaptive", "gray"]
    chosen_img = None

    for name in preferred_order:
        for variant_name, variant_img in variants:
            if variant_name == name:
                chosen_img = variant_img
                break
        if chosen_img:
            break

    if chosen_img is None:
        chosen_img = variants[0][1]

    name, _ = os.path.splitext(save_base_path)
    processed_path = name + "_processed.jpg"
    chosen_img.save(processed_path, "JPEG", quality=95)
    return processed_path

# ---------------- OCR EXTRACTION ----------------
def extract_text_from_single_image(image_path):
    variants = generate_preprocessed_variants(image_path)

    best_text = ""
    best_score = -1

    for _, variant_img in variants:
        text, score = run_tesseract_best(variant_img)
        if score > best_score:
            best_score = score
            best_text = text

    processed_path = save_best_processed_variant(variants, image_path)
    return best_text, processed_path

def extract_text(file_path):
    full_text = []
    processed_file = None

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=200)

        for i, page in enumerate(pages):
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_page_{i}.jpg")
            page.save(temp_path, "JPEG")

            page_text, processed_path = extract_text_from_single_image(temp_path)
            full_text.append(page_text)

            if i == 0 and processed_path:
                processed_file = processed_path

            if os.path.exists(temp_path):
                os.remove(temp_path)

    else:
        text, processed_path = extract_text_from_single_image(file_path)
        full_text.append(text)
        processed_file = processed_path

    merged_text = "\n".join([t for t in full_text if t.strip()])
    return merged_text, processed_file

# ---------------- NORMALIZATION ----------------
def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")

    text = re.sub(r"\|", "I", text)
    text = re.sub(r"(?<=\b)0(?=[A-Za-z])", "O", text)
    text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", text)
    text = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", text)
    text = re.sub(r"(?<=[A-Za-z])5(?=[A-Za-z])", "s", text)

    text = re.sub(r"[^A-Za-z0-9\s.,;:!?'\-()/%]", " ", text)
    text = re.sub(r"([.,;:!?]){2,}", r"\1", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)

    return text.strip()

# ---------------- SPELL CORRECTION ----------------
def correct_word(word):
    if not SPELLCHECK_AVAILABLE:
        return word

    prefix_match = re.match(r"^\W+", word)
    suffix_match = re.search(r"\W+$", word)

    prefix = prefix_match.group(0) if prefix_match else ""
    suffix = suffix_match.group(0) if suffix_match else ""

    core = re.sub(r"^\W+|\W+$", "", word)

    if not core:
        return word

    if len(core) <= 3 or any(ch.isdigit() for ch in core):
        return word

    lower_core = core.lower()

    if lower_core in spell:
        return word

    corrected = spell.correction(lower_core)
    if not corrected:
        return word

    if abs(len(corrected) - len(core)) > 2:
        return word

    if core.isupper():
        corrected = corrected.upper()
    elif core.istitle():
        corrected = corrected.title()

    return prefix + corrected + suffix

def correct_text(text):
    if not text:
        return ""

    if not SPELLCHECK_AVAILABLE:
        return text

    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    corrected_text = " ".join(corrected_words)

    corrected_text = re.sub(r"\s+([.,;:!?])", r"\1", corrected_text)
    corrected_text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", corrected_text)

    return corrected_text.strip()

# ---------------- AUTOMATIC OLLAMA CORRECTION ----------------
def correct_text_with_ollama(raw_text):
    """
    Automatically correct OCR output using Ollama.
    """
    prompt = f"""
You are an OCR text correction assistant.

The following text was extracted from a handwritten image or PDF using OCR.
Correct spelling mistakes, broken words, punctuation, and grammar.
Keep the original meaning unchanged.
Do not explain anything.
Return only the corrected text.

OCR Text:
{raw_text}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45)
        if response.status_code == 200:
            data = response.json()
            corrected = data.get("message", {}).get("content", raw_text).strip()
            return corrected if corrected else raw_text
        return raw_text
    except Exception:
        return raw_text

# ---------------- FULL POSTPROCESS ----------------
def postprocess_text(raw_text):
    """
    Always:
    1. normalize OCR text
    2. apply basic spell correction
    3. send to Ollama for final correction
    4. normalize again
    """
    normalized = normalize_text(raw_text)
    corrected = correct_text(normalized)

    if corrected.strip():
        corrected = correct_text_with_ollama(corrected)

    final_text = normalize_text(corrected)
    return final_text

# ---------------- MARK / FEEDBACK SPLIT ----------------
def parse_evaluation_result(evaluation_text):
    score = ""
    feedback = evaluation_text.strip()

    score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?\/10)", evaluation_text, re.IGNORECASE)
    feedback_match = re.search(r"Feedback:\s*(.*)", evaluation_text, re.IGNORECASE | re.DOTALL)

    if score_match:
        score = score_match.group(1).strip()

    if feedback_match:
        feedback = feedback_match.group(1).strip()

    return score, feedback

# ---------------- OLLAMA EVALUATION ----------------
def evaluate_with_ollama(student, reference, question):
    prompt = f"""
Question: {question}
Reference Answer: {reference}
Student Answer: {student}

Evaluate the student's answer based on correctness, relevance, and completeness.

Return exactly in this format:
Score: <number>/10
Feedback: <short feedback>
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "No response received.")
        return f"Ollama error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

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

                raw_text, processed_path = extract_text(file_path)

                # Always auto-correct with Ollama
                extracted_text = postprocess_text(raw_text)

                if processed_path:
                    processed_image = os.path.basename(processed_path)
            else:
                extracted_text = "No valid file selected."

        # -------- EVALUATE --------
        elif "evaluate" in request.form:
            if extracted_text.strip() and question.strip() and reference.strip():
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
        feedback=feedback
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)