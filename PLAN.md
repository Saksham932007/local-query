# Hackathon Plan: "LocalQuery"

## 1. Project Goal

* **MVP (Minimum Viable Product):** A local web app where a user can upload a text or PDF document, ask a question, and get an answer extracted from that document using a *pre-trained* Transformer model.
* **Stretch Goal (The "Hack"):** Fine-tune a smaller Transformer model (like `DistilBERT` or `MiniLM`) on a domain-specific Q&A dataset (e.g., SQuAD, or something more specific if you can find it).
* **Demo:** Show the app answering a question from an uploaded PDF. Then, (if you hit the stretch goal) show a *better* answer using your fine-tuned model vs. the base model.

## 2. Tech Stack

* **Language:** Python 3.9+
* **Environment:** `venv`
* **Core NLP:** `transformers`, `torch` (or `tensorflow`)
* **App Framework:** `streamlit` (This is our hackathon secret weapon. No HTML/JS.)
* **Doc Parsing:** `PyMuPDF` (for PDFs), `python-docx` (for .docx)
* **Dataset (for stretch goal):** `datasets` (to load SQuAD or other data)

## 3. The 48-Hour Battle Plan

### Phase 0: Setup (Est: 1-2 Hours)

1.  **Init Project:**
    * `git init local-query`
    * `cd local-query`
    * Copy in the `.gitignore` file.
2.  **Create Environment:**
    * `python3 -m venv venv`
    * `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3.  **Install Dependencies:**
    * `pip install -r requirements.txt`
4.  **Test Installation:**
    * Run `streamlit run app.py`. You should see the skeleton app load in your browser.

### Phase 1: The Core MVP (Est: 4-6 Hours)

* **Goal:** Make `app.py` fully functional with a *pre-trained* model.
* **File:** `app.py`
... (rest of plan) ...