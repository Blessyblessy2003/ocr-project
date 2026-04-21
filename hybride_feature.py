import re

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- TOKENIZATION ----------------
def tokenize(text):
    text = clean_text(text)
    return text.split()

# ---------------- SEMANTIC FEATURE ----------------
# keyword matching between reference answer and student answer
def keyword_matching(reference_text, student_text):
    reference_words = set(tokenize(reference_text))
    student_words = set(tokenize(student_text))

    if not reference_words:
        return 0, [], []

    matched_keywords = reference_words.intersection(student_words)
    unmatched_keywords = reference_words.difference(student_words)

    keyword_score = len(matched_keywords) / len(reference_words)

    return keyword_score, list(matched_keywords), list(unmatched_keywords)

# ---------------- LEXICAL FEATURE ----------------
# correct word matching between reference answer and student answer
def correct_word_matching(reference_text, student_text):
    reference_words = tokenize(reference_text)
    student_words = tokenize(student_text)

    matched_words = []
    unmatched_words = []

    ref_count = {}

    for word in reference_words:
        ref_count[word] = ref_count.get(word, 0) + 1

    stu_count = {}
    for word in student_words:
        stu_count[word] = stu_count.get(word, 0) + 1

    total_correct_matches = 0

    for word in ref_count:
        if word in stu_count:
            match_count = min(ref_count[word], stu_count[word])
            total_correct_matches += match_count
            matched_words.extend([word] * match_count)

            if ref_count[word] > match_count:
                unmatched_words.extend([word] * (ref_count[word] - match_count))
        else:
            unmatched_words.extend([word] * ref_count[word])

    total_reference_words = len(reference_words)
    lexical_score = total_correct_matches / total_reference_words if total_reference_words > 0 else 0

    return lexical_score, matched_words, unmatched_words

# ---------------- HYBRID FEATURE EXTRACTION ----------------
def extract_hybrid_features(reference_text, student_text):
    semantic_score, matched_keywords, unmatched_keywords = keyword_matching(reference_text, student_text)
    lexical_score, matched_words, unmatched_words = correct_word_matching(reference_text, student_text)

    return {
        "semantic_score": semantic_score,
        "matched_keywords": matched_keywords,
        "unmatched_keywords": unmatched_keywords,
        "lexical_score": lexical_score,
        "matched_words": matched_words,
        "unmatched_words": unmatched_words
    }