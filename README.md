# Sentiment & Sarcasm-Aware Tone Shifter

A Python project that analyzes sentiment, detects sarcasm, and rewrites text with tone adjustments.  

---

## Features
- **Sentiment Analysis**: Classifies text as Positive, Negative, or Neutral with confidence scores.  
- **Sarcasm Detection**: Identifies sarcasm using mixed sentiment cues, keywords, and emojis.  
- **Tone Shifting**: 
  - Softens negative words (e.g., "hate" → "don’t prefer").  
  - Intensifies positive expressions (e.g., "good" → "great").  
- **Interactive CLI**: Analyze sentences in real-time with sentiment report and rewritten output.

---

## Tech Stack
- Python 3.9+  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)  
- PyTorch  
- `emoji` (for emoji handling)  
- Regex for sarcasm keyword detection  

---

## License

MIT License


