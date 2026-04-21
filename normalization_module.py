import re

try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    SPELLCHECK_AVAILABLE = True
except Exception:
    SPELLCHECK_AVAILABLE = False
    spell = None

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