# 🔥 Advanced Logic Contradiction Detector

## 🧠 Revolutionary Tool for Analyzing Thinking Quality

This project combines two unique analyzers:
1. **Rationality Analyzer** — measures emotional content in text
2. **Logic Contradiction Detector** — finds errors in reasoning

---

## ✨ Features

### 🎯 What this tool does:

- **Measures text rationality** (0-100%)
  - Detects emotional markers
  - Counts logical connectors and scientific terms
  - Analyzes sentence structure

- **Finds 6 types of contradictions:**
  1. **Direct contradictions** — opposite statements in one sentence
  2. **Self-contradictions** — weakening statements with "but", "however"
  3. **Hidden contradictions** — contradictions in phrase structure
  4. **Semantic contradictions** — meaning-based contradictions (via NLI model)
  5. **Temporal contradictions** — time-related logical errors
  6. **Quantitative contradictions** — quantity-related logical errors

- **Evaluates confidence and severity**
  - Each contradiction gets confidence score (0-100%)
  - Classifies by severity level (high/medium/low)
  - Calculates impact score for each finding

- **Provides detailed recommendations**
  - Explains how to improve text
  - Shows weak points in reasoning
  - Gives overall quality assessment

---

## 🚀 Installation

### Requirements:
- Python 3.8+
- 4+ GB RAM (for NLI model loading)
- Internet connection (for first model download)
- Modern web browser (Chrome, Firefox, Edge)

### Steps:

1. **Clone repository:**
```bash
git clone https://github.com/kris-stern/advanced-logic-detector.git
cd advanced-logic-detector