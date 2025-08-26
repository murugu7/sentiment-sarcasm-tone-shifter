import torch
import warnings
import re
import emoji
from transformers import pipeline

# Silence warnings to keep console clean
warnings.filterwarnings("ignore")

DEVICE = 0 if torch.cuda.is_available() else -1

# Hugging‑Face sentiment model (small & fast)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment = pipeline(
    "sentiment-analysis",
    model=MODEL,
    tokenizer=MODEL,
    top_k=None,              # get all label scores
    device=DEVICE,
    framework="pt",
)

# Map model label IDs to meaningful names
LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# Simple lexical cues & emojis for sarcasm
SARC_KEYWORDS = [
    r"\byeah\s+right\b",
    r"\bsure\b",
    r"\bas\s+if\b",
    r"\boh\b.*\b(great|nice|wonderful)\b",
]
SARC_EMOJIS = [
    ":face_with_rolling_eyes:",
    ":smirking_face:",
    ":unamused_face:",
    ":face_with_raised_eyebrow:",
]

# Tone‑shift dictionaries
NEG2_SOFT = {
    "hate": "don’t prefer",
    "worst": "not the best",
    "terrible": "could be better",
    "awful": "less than ideal",
    "disappointed": "not fully satisfied",
    "useless": "not very helpful",
    "stupid": "unwise",
    "dumb": "unwise",
    "boring": "not very exciting",
    "bad": "not great",
    "problem": "issue",
    "slow": "not very fast",
    "broke": "stopped working",
    "didn't work": "wasn't functional",
    "not working": "currently unavailable"
}
POS2_INTENSE = {
    "good": "great",
    "great": "excellent",
    "nice": "wonderful",
    "like": "really like",
    "love": "absolutely love",
    "interesting": "fascinating",
    "happy": "delighted",
    "cool": "awesome",
}

# ---------------- Sarcasm detector (heuristic) ---------------- #
def detect_sarcasm(text: str, scores) -> bool:
    pos = next(d["score"] for d in scores if d["label"] == "LABEL_2")
    neg = next(d["score"] for d in scores if d["label"] == "LABEL_0")

    mixed_polarity = pos >= 0.35 and neg >= 0.35
    text_lower = text.lower()
    keyword_hit = any(re.search(pat, text_lower) for pat in SARC_KEYWORDS)
    emoji_hit = any(e in text for e in SARC_EMOJIS)

    return (mixed_polarity and (keyword_hit or emoji_hit)) or keyword_hit

# ---------------- Tone‑shift paraphraser ---------------- #
def positive_paraphrase(text: str, sentiment_label: str) -> str:
    """Soften negative words or intensify positive wording."""
    new_text = text

    # Apply negative‑to‑soft replacements
    for bad, soft in NEG2_SOFT.items():
        new_text = re.sub(rf"\b{re.escape(bad)}\b", soft, new_text, flags=re.IGNORECASE)

    # If original sentiment is Neutral or Positive, also intensify positive wording
    if sentiment_label in {"Neutral", "Positive"}:
        for pos, intense in POS2_INTENSE.items():
            new_text = re.sub(rf"\b{re.escape(pos)}\b", intense, new_text, flags=re.IGNORECASE)

    # Return new text only if something changed
    return new_text if new_text != text else "(no tone change needed)"

# ---------------- Main analysis ---------------- #
def analyze(text: str):
    original = text
    text_demojized = emoji.demojize(original)

    scores = sentiment(text_demojized)[0]
    top_label_id = max(scores, key=lambda x: x["score"])["label"]
    readable_label = LABEL_MAP[top_label_id]

    sarcastic = detect_sarcasm(text_demojized, scores)
    rewrite = positive_paraphrase(original, readable_label)

    print("\n🧠 Sentiment Report")
    print(f"Text      : {emoji.emojize(original)}")
    print(f"Sentiment : {readable_label}")
    print("Confidence: " + ", ".join(f"{LABEL_MAP[s['label']]} = {s['score']:.2f}" for s in scores))
    print(f"Sarcasm   : {'⚠️  Likely sarcastic' if sarcastic else '—'}")
    print(f"Rewrite   : ✨ {rewrite}")
    print("-" * 40)

# ---------------- CLI loop ---------------- #
if __name__ == "__main__":
    print("Type a sentence to analyze (type 'exit' to quit):\n")
    while True:
        user_input = input("► ")
        if user_input.lower() == "exit":
            break
        analyze(user_input)
