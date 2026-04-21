import re

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