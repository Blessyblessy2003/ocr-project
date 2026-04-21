import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path

def resize_for_ocr(pil_img, scale=1.5):
    w, h = pil_img.size
    return pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

def cv_to_pil(cv_img):
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def generate_preprocessed_variants(image_path):
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

def extract_text(file_path, upload_folder):
    full_text = []
    processed_file = None

    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=200)

        for i, page in enumerate(pages):
            temp_path = os.path.join(upload_folder, f"temp_page_{i}.jpg")
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