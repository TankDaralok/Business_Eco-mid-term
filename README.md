Business Essentials Quiz — Chapters 1–6
======================================

Overview
--------
This project extracts questions from a textbook PDF and a test-bank PDF, maps the questions to chapters 1–6, classifies question types, and provides a Streamlit web app to run quizzes, retry wrong answers, save results, and export CSV.

Files
-----
- extract_questions.py — extraction/module script (contains unit-testable functions)
- quiz_app.py — Streamlit application (primary app)
- data/questions_ch1_6.json — sample dataset (3 example questions)
- logs/extraction_log.txt — extraction log (generated)
- logs/maybe_unmatched.json — ambiguous questions (generated)
- results/ — output folder for quiz results (generated)

Installation
------------
1. Create a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # mac/linux
   venv\Scripts\activate     # windows

2. Install dependencies:
   pip install streamlit pdfplumber rapidfuzz pandas tqdm python-magic

3. Place your local PDFs in the project folder:
   - Business_Essentials_13th_Ed.pdf
   - Stuvia-7183014-test-bank-for-Business-Essentials13th-edition.pdf

   OR upload them via the Streamlit sidebar to run extraction.

How to run
----------
Run the Streamlit app:
  streamlit run quiz_app.py

Extraction workflow
-------------------
- If you have the PDFs, you can:
  - Run extraction from the command line (quick):
    python extract_questions.py --book Business_Essentials_13th_Ed.pdf --testbank Stuvia-7183014-test-bank-for-Business-Essentials13th-edition.pdf --out data/questions_ch1_6.json --threshold 70

  - Or open the Streamlit app, upload both files in the sidebar, pick a fuzzy threshold, and click "Run extraction".

- The script:
  - Extracts text from PDFs using pdfplumber (fast and reliable for text-based PDFs).
  - Detects chapter headings using multiple regex heuristics (Chapter 1, CHAPTER 1, Ch.1, "1 The ...").
  - Builds a mapping for chapters 1–6: start/end char indices and chapter text.

Matching heuristic
------------------
- Preferred: direct chapter labels in the test bank (e.g., "Chapter 3").
- Fallback: fuzzy semantic matching using rapidfuzz.partial_ratio
  - Compares each question block (first ~1000 chars) to the chapter title and the first ~2000 chars of chapter text.
  - Uses a tunable threshold parameter (default 70).

- Tuning:
  - Higher threshold (85-95): more conservative — fewer false matches, more "maybe_unmatched".
  - Lower threshold (60-70): more permissive — may match ambiguous questions.
  - Change in Streamlit sidebar or command-line --threshold.

Question parsing & classification
---------------------------------
- parse_testbank_text: splits the test-bank text into blocks using patterns like:
  - lines starting with "Q1", "Question 1", "1)", "1." etc.
  - fallback splitting by blank lines.
- classify_question uses regex heuristics to detect:
  - multiple_choice: presence of "A./A) B./B) C./C)" patterns
  - true_false: explicit True/False markers
  - fill_in_blank: underscores "____"
  - short_answer: fallback when ends with "?" or is short with no options
  - unknown: needs manual review

Output schema (data/questions_ch1_6.json)
-----------------------------------------
Each entry:
{
  "chapter": 1,
  "type": "multiple_choice",
  "question": "Full question text ...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B",
  "explanation": "Optional"
}

App features
------------
- Quiz composition settings (sidebar): number of questions, per-type preferences, ensure at least one per chapter, randomize order.
- Progress bar and per-question UI:
  - MCQs use radio buttons; T/F are radios; short answers are text inputs.
- Results:
  - Score by type and overall.
  - Review list of incorrect questions (shows correct answers if known).
  - Retry wrong questions.
  - Save results to results/<timestamp>_results.json.
  - Export CSV.

Robustness & logs
-----------------
- Ambiguous matches are recorded in logs/maybe_unmatched.json and logs/extraction_log.txt.
- Debug mode in Streamlit exposes these logs and offers downloads.
- Missing answer keys or malformed options are handled: question classified as 'unknown' and logged for manual review.

Sample run (without PDFs)
-------------------------
The repository includes data/questions_ch1_6.json (3 example questions). Launch the app:
  streamlit run quiz_app.py

You can run a small quiz (10 questions default will be reduced to available 3 sample items).

Testing & validation
--------------------
Unit-testable functions in extract_questions.py:
- extract_text_from_pdf(path)  -- returns str (for .txt you can call open().read())
- detect_chapters(book_text)
- parse_testbank_text(tb_text)
- map_question_to_chapter(question_text, chapter_texts, threshold)
- classify_question(block_text)

If you'd like, you can write pytest tests around these functions using the sample JSON / synthetic inputs.

Copyright & ethical note
------------------------
- This tool only operates on local files you provide. Do NOT use it to fetch or scrape online content.
- Respect copyright — only process files that you are authorized to use.

Contact / next steps
--------------------
If you want:
- Improved NLP mapping (use sentence-transformers embeddings + cosine similarity instead of rapidfuzz).
- Better extraction for scanned PDFs (requires OCR; pytesseract).
- A database instead of JSON for large-scale quizzes.

