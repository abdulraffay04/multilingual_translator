# Multilingual Neural Translator (NLLB-1.3B)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![Model](https://img.shields.io/badge/Model-NLLB_200_1.3B-green)
![License](https://img.shields.io/badge/License-MIT-orange)

A **High-Fidelity Neural Machine Translation (NMT)** web application built with **Flask** and **PyTorch**, powered by Meta AI's **NLLB-200 (1.3 Billion Parameter)** model.

This project implements a **Hybrid NMT Architecture**, combining state-of-the-art Deep Learning with a custom **Rule-Based Glossary** to solve common translation errors in low-resource languages like **Urdu** and **Arabic**.

<img width="1918" height="1013" alt="a4" src="https://github.com/user-attachments/assets/2dc0dd83-0048-453c-b425-bfe2ead9806b" />

---

## Key Features

* **Advanced AI Engine:** Runs the **1.3 Billion Parameter** NLLB-200 model locally for superior grammatical accuracy compared to standard lightweight APIs.
* **Hybrid Glossary Pipeline:** A custom post-processing layer that fixes literal idiom translations (e.g., translates *"Piece of cake"* contextually as *"Easy task"* instead of *"Slice of cake"*).
* **Smart Grammar Handling:** Correctly handles **Subject-Object-Verb (SOV)** sentence structures for Urdu, preventing broken sentence ordering.
* **Auto-RTL Support:** The UI automatically switches text direction to **Right-to-Left** when Urdu or Arabic is detected.
* **Auto Language Detection:** Uses statistical N-gram analysis (`langdetect`) to identify the source language instantly.

---

## The Challenge & Solution

### The Problem: Standard AI Failures
Most generic translation tools translate word-for-word.
1.  **Broken Grammar:** They fail to reorder English (SVO) sentences into Urdu (SOV).
    * *Input:* "The doctor **who treated my brother**..."
    * *Bad Output:* "Doctor **kaun ilaj kiya**..." (Broken flow)
2.  **Lost Idioms:** They kill cultural nuances.
    * *Input:* "It was a **piece of cake**."
    * *Bad Output:* "It was a **cake's slice**." (Literal meaning)

### My Solution: Hybrid Architecture
I engineered a pipeline that intervenes in the translation process:
1.  **Deep Learning:** Uses the massive 1.3B parameter model to handle complex grammar reordering.
2.  **Rule-Based Logic:** A Python dictionary scans the output and hot-swaps literal mistakes with cultural proverbs.

**Result:** *"Piece of cake"* $\rightarrow$ *"بائیں ہاتھ کا کھیل"* (Left hand's game / Easy task).

---

## The NLP Pipeline (Technical Workflow)

The system processes text through a 6-step pipeline:

1.  **Language Identification (LID):**
    * Input text is analyzed using `langdetect` to predict the source language code (e.g., `ur`, `ar`, `en`).
2.  **Tokenization:**
    * Text is converted into sub-word tokens using the `NllbTokenizer`.
    * Special language tags (e.g., `__urd_Arab__`) are injected to guide the model.
3.  **Neural Inference (Seq2Seq):**
    * The **Transformer Model** processes tokens on the CPU/GPU.
    * **Beam Search (k=5):** Explores 5 simultaneous sentence paths to find the most grammatically correct output.
    * **Repetition Penalty:** Applied to prevent loop errors (e.g., "Hello Hello Hello").
4.  **Decoding:**
    * Tensor IDs are mapped back into human-readable text strings.
5.  **Hybrid Glossary Check (Post-Processing):**
    * The system checks the source text for known idioms. If found, it overrides the AI's literal translation with a pre-defined correct phrase.
6.  **Frontend Rendering:**
    * The API returns the JSON response.
    * JavaScript detects if the target is Urdu/Arabic and applies `direction: rtl` CSS styles dynamically.

---

## Tech Stack

### Backend
* **Language:** Python 3.9+
* **Framework:** Flask (Micro-framework)
* **AI Engine:** PyTorch, Hugging Face Transformers
* **Utilities:** Flask-CORS, LangDetect

### Frontend
* **Core:** HTML5, CSS3, JavaScript (ES6)
* **Design:** CSS Glassmorphism & Flexbox
* **Icons:** FontAwesome

---



https://github.com/user-attachments/assets/8dec4b2e-2c1f-42a3-8c1b-0936c5127942


## ABDUL RAFFAY
[LinkedIn](https://www.linkedin.com/in/abdul-raffay-b1b1a62aa/)
