import requests
from normalization_module import normalize_text, correct_text

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral:latest"

def correct_text_with_ollama(raw_text):
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

def postprocess_text(raw_text):
    normalized = normalize_text(raw_text)
    corrected = correct_text(normalized)

    if corrected.strip():
        corrected = correct_text_with_ollama(corrected)

    final_text = normalize_text(corrected)
    return final_text

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