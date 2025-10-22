"""
extract_questions.py

Helpers for extracting book and test-bank text, detecting chapter ranges (1-6),
mapping questions to chapters (direct label OR fuzzy match), classifying question types,
and exporting filtered questions to JSON. Logs ambiguous / maybe-unmatched items.

Functions (unit-testable):
- extract_text_from_pdf(path) -> str
- detect_chapters(book_text, chapters_to_find=(1,2,3,4,5,6)) -> dict[int, (start_idx, end_idx, title, text)]
- parse_testbank_text(tb_text) -> List[dict]  # naive question splitter
- map_question_to_chapter(question_text, chapter_texts, threshold=70) -> int | None
- classify_question(question_text, options=[]) -> str
- build_questions_dataset(book_path, testbank_path, out_json_path, ...) -> dict (summary)

Author: ChatGPT (as requested)
"""

from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pdfplumber
from rapidfuzz import fuzz
from tqdm import tqdm

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "extraction_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# --- Utility: extract text from PDF ---
def extract_text_from_pdf(path: str) -> str:
    """
    Extract plain text from a PDF file using pdfplumber.
    Returns the concatenated text of all pages.
    """
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                ptext = page.extract_text() or ""
            except Exception as e:
                logging.warning("pdfplumber page.extract_text error: %s", e)
                ptext = ""
            text_parts.append(ptext)
    return "\n\n".join(text_parts)

# --- Detect chapter headings robustly ---
CHAPTER_HDR_PATTERNS = [
    r"^\s*Chapter\s+(\d{1,2})\b",            # Chapter 1
    r"^\s*CHAPTER\s+(\d{1,2})\b",            # CHAPTER 1
    r"^\s*Ch\.\s*(\d{1,2})\b",               # Ch. 1
    r"^\s*Chapter\s+One\b",                  # "Chapter One" (rare)
    r"^\s*CH\.\s*(\d{1,2})\b",
    r"^\s*(\d{1,2})\s+The\s+",               # "1 The ..." (sometimes)
]

def detect_chapters(book_text: str, chapters_to_find: Tuple[int, ...] = (1,2,3,4,5,6)) -> Dict[int, Dict]:
    """
    Detect approximate text spans for requested chapters.
    Returns dict mapping chapter_number -> {'start': idx, 'end': idx, 'title': str, 'text': str}
    Heuristic:
    - Scan lines for chapter header regexes and record their character index.
    - For headings that include number, map to chapter number.
    - If fewer headers found, do fuzzy match on "Chapter {n}" in whole text.
    - end index = next chapter start or end of book.
    """
    chapters_to_find = tuple(chapters_to_find)
    idxs = []  # list of tuples (chapter_number, start_char_index, heading_line)
    # iterate through text lines to find headings with char index
    for match in re.finditer(r"(?m)^.*$", book_text):
        line = match.group(0)
        start = match.start()
        # try matching each pattern
        for pat in CHAPTER_HDR_PATTERNS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                # attempt to extract number
                try:
                    num = int(m.group(1))
                except Exception:
                    # if "Chapter One" etc, try to map words to numbers (not exhaustive)
                    word = line.strip().split()
                    num = None
                    if len(word) >= 2 and word[0].lower().startswith("chapter"):
                        wn = word[1].lower()
                        WORD_NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}
                        num = WORD_NUM.get(wn)
                if num and num in chapters_to_find:
                    idxs.append((num, start, line.strip()))
                    logging.debug("Detected header for chapter %s at %d: %s", num, start, line.strip())
                break

    # If not found for all chapters, attempt fuzzy search for "Chapter {n}"
    found_nums = {t[0] for t in idxs}
    for n in chapters_to_find:
        if n in found_nums:
            continue
        # fuzzy search for "chapter n" anywhere: use rapidfuzz partial_ratio
        target = f"chapter {n}"
        best_score = 0
        best_pos = None
        # examine windows (first 20000 chars for performance)
        text_sample = book_text[:500000]  # first half-million chars
        for m in re.finditer(r"(?i)chapter[\s\._-]*\w{1,10}", text_sample):
            cand = m.group(0)
            score = fuzz.partial_ratio(target, cand)
            if score > best_score:
                best_score = score
                best_pos = m.start()
        if best_score >= 80 and best_pos is not None:
            idxs.append((n, best_pos, f"fuzzy:{target} (score={best_score})"))
            logging.info("Fuzzy-located chapter %d at pos %d (score=%d)", n, best_pos, best_score)

    # assemble mapping
    # sort by start index
    idxs_sorted = sorted(idxs, key=lambda x: x[1])
    chapter_map: Dict[int, Dict] = {}
    for i, (num, start, heading) in enumerate(idxs_sorted):
        # determine end: next start -1 or end of text
        if i+1 < len(idxs_sorted):
            end = idxs_sorted[i+1][1]
        else:
            end = len(book_text)
        chapter_text = book_text[start:end].strip()
        chapter_map[num] = {"start": start, "end": end, "title": heading, "text": chapter_text}
        logging.info("Chapter %d mapped start=%d end=%d title=%s", num, start, end, heading)
    return chapter_map

# --- Parse test bank into question blocks ---
def parse_testbank_text(tb_text: str) -> List[Dict]:
    """
    Parse test bank text into a list of question dicts.
    This is heuristic: looks for lines that start with a number or with letters like 'Q1.' or '1)'.
    Returns list of {'raw': full_text_block, 'short': first_line}
    """
    # Normalize line endings
    lines = tb_text.splitlines()
    q_blocks = []
    current_lines = []
    # pattern to detect question starter:
    starter_re = re.compile(r'^\s*((Q|Question)\s*\d+|(\d{1,3})[.)\s])\s*', re.IGNORECASE)
    for line in lines:
        if starter_re.match(line):
            # start a new block
            if current_lines:
                q_blocks.append({"raw": "\n".join(current_lines).strip()})
            current_lines = [line]
        else:
            if current_lines:
                current_lines.append(line)
            else:
                # stray text before first question
                continue
    if current_lines:
        q_blocks.append({"raw": "\n".join(current_lines).strip()})
    # If no starters found, fallback: split by blank lines and treat each as potential question
    if not q_blocks:
        parts = re.split(r'\n\s*\n', tb_text)
        for p in parts:
            p = p.strip()
            if p:
                q_blocks.append({"raw": p})
    # attach a short preview
    for q in q_blocks:
        q["short"] = q["raw"].splitlines()[0][:200]
    logging.info("Parsed %d question blocks from test bank", len(q_blocks))
    return q_blocks

# --- Try to extract explicit chapter label from a question block ---
CH_LABEL_RE = re.compile(r'chapter\s*(\d{1,2})', re.IGNORECASE)
def extract_chapter_label_from_block(block_text: str) -> Optional[int]:
    m = CH_LABEL_RE.search(block_text)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

# --- Map question to chapter using direct label OR fuzzy match ---
def map_question_to_chapter(question_text: str, chapter_texts: Dict[int, Dict], threshold: int = 70) -> Optional[int]:
    """
    Map a question to a chapter.
    - First: try to find an explicit label in the question (e.g., 'Chapter 3').
    - Else: for each candidate chapter (1..6), compute similarity between the question and
      either (chapter title) or (first 400-800 chars of chapter text). Use rapidfuzz's ratio.
    - If best score >= threshold -> return that chapter, else return None.
    Parameters:
      question_text: full text of question block
      chapter_texts: mapping from chapter_number to {'title','text',...}
      threshold: int (0-100) similarity threshold. Default 70 (tunable).
    """
    # 1) direct label
    lab = extract_chapter_label_from_block(question_text)
    if lab and lab in chapter_texts:
        logging.debug("Direct label found chapter %d in question", lab)
        return lab

    # 2) fuzzy match: compare against chapter first N chars and title
    best_score = 0
    best_ch = None
    question_snippet = " ".join(question_text.split())[:1000]  # limit
    for ch_num, ch_info in chapter_texts.items():
        # compare to title first
        title = ch_info.get("title", "")
        score_title = fuzz.partial_ratio(question_snippet, title)
        # compare to first chunk of chapter text
        chapter_snippet = ch_info.get("text", "")[:2000]  # first ~2000 chars
        score_body = fuzz.partial_ratio(question_snippet, chapter_snippet)
        # combine heuristically: prefer body match
        score = max(score_body, score_title)
        if score > best_score:
            best_score = score
            best_ch = ch_num
    logging.debug("Mapping best candidate chapter=%s score=%d", best_ch, best_score)
    if best_score >= threshold:
        return best_ch
    else:
        logging.info("Low confidence mapping (best_score=%d) for question: %s", best_score, question_text[:120])
        return None

# --- Classify question types ---
MULT_CHOICE_RE = re.compile(r'(^|\n)\s*([A-D]|[1-4])[\.\)]\s+', re.IGNORECASE)
TF_RE = re.compile(r'\b(True|False|T/F|T or F)\b', re.IGNORECASE)
BLANK_RE = re.compile(r'_{3,}|_{1,}\bBlank\b|_____', re.IGNORECASE)
# Simple pattern to detect answer keys like 'Answer: A' or 'Answer: 3' within block
ANSWER_KEY_RE = re.compile(r'Answer[:\s]*([A-D]|[A-D]\b|[1-9][0-9]?|True|False|T|F)', re.IGNORECASE)

def extract_options_from_block(block_text: str) -> List[str]:
    """
    Attempt to extract option lines (A. ..., B. ...) from a question block.
    Returns list of option strings in order.
    """
    opts = []
    # find lines that start with A. B. or A) B) or 1) etc.
    for m in re.finditer(r'(?m)^\s*([A-D]|[1-9][0-9]?)\s*[\.\)]\s*(.+)

2) quiz_app.py
Save as quiz_app.py
```python
"""
quiz_app.py

Streamlit quiz app that:
- Loads data/questions_ch1_6.json (or runs extraction if user uploads files)
- Allows settings: how many questions per type, randomization, ensure at least one per chapter, debug mode
- Runs quiz with MCQ / T/F / short answer / fill-in / unknown
- Shows progress bar, saves responses to session_state
- Results page shows per-type and overall scores, lists incorrect questions with explanations
- Supports "Retry Wrong Questions" to re-run only wrong ones
- Persists results to results/<timestamp>_results.json and supports CSV export

Usage:
  streamlit run quiz_app.py
"""

from __future__ import annotations
import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import random
import pandas as pd
from typing import List, Dict, Any, Optional
import extract_questions as eqmod  # expects extract_questions.py in same folder

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Business Essentials Quiz (Ch 1-6)", layout="wide")

# --- Helper UI / load functions ---
def load_questions(path: Path = DATA_DIR / "questions_ch1_6.json") -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(result_obj: Dict[str, Any]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{ts}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False)
    return out_path

def to_csv_string(results: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(results)
    return df.to_csv(index=False)

# --- Sidebar: extraction or file upload ---
st.sidebar.title("Quiz Settings & Data")
st.sidebar.markdown("Upload textbook/testbank PDFs (optional) to run extraction, or use existing data/questions_ch1_6.json")

uploaded_book = st.sidebar.file_uploader("Upload textbook PDF (.pdf) or .txt", type=["pdf","txt"])
uploaded_tb = st.sidebar.file_uploader("Upload testbank PDF (.pdf) or .txt", type=["pdf","txt"])

fuzzy_threshold = st.sidebar.slider("Fuzzy matching threshold", min_value=50, max_value=95, value=70, step=1)
debug_mode = st.sidebar.checkbox("Debug mode (show logs & ambiguous matches)", value=False)

run_extraction = st.sidebar.button("Run extraction (book + testbank)")

if run_extraction:
    if uploaded_book is None or uploaded_tb is None:
        st.sidebar.error("Please upload both textbook and testbank files to run extraction.")
    else:
        # Save uploaded files to temp and run extraction
        book_path = Path("temp_book_upload.pdf")
        tb_path = Path("temp_tb_upload.pdf")
        # write files
        with open(book_path, "wb") as f:
            f.write(uploaded_book.getbuffer())
        with open(tb_path, "wb") as f:
            f.write(uploaded_tb.getbuffer())
        st.sidebar.info("Running extraction (this may take a while)...")
        summary = eqmod.build_questions_dataset(str(book_path), str(tb_path), out_json_path=str(DATA_DIR / "questions_ch1_6.json"), fuzzy_threshold=fuzzy_threshold)
        st.sidebar.success(f"Extraction done. Matched: {summary['matched_to_ch1_6']} questions.")
        if debug_mode:
            st.write("Extraction summary:", summary)
            # show maybe_unmatched
            maybe_file = LOG_DIR / "maybe_unmatched.json"
            if maybe_file.exists():
                maybe = json.load(open(maybe_file, "r", encoding="utf-8"))
                st.write("Maybe unmatched sample (first 10):", maybe[:10])

# load questions
questions = load_questions()
st.sidebar.markdown(f"Questions loaded: {len(questions)} from data/questions_ch1_6.json")

# settings for quiz composition
st.sidebar.header("Quiz Composition")
total_questions = st.sidebar.number_input("Total number of questions to draw (0 = use all)", min_value=0, max_value=200, value=10, step=1)
per_type_mc = st.sidebar.number_input("MCQ count (preferred)", min_value=0, max_value=200, value=6, step=1)
per_type_tf = st.sidebar.number_input("True/False count (preferred)", min_value=0, max_value=200, value=2, step=1)
per_type_sa = st.sidebar.number_input("Short answer count (preferred)", min_value=0, max_value=200, value=2, step=1)
ensure_one_per_chapter = st.sidebar.checkbox("Guarantee at least one question per chapter (1-6)", value=False)
randomize_by_chapter = st.sidebar.checkbox("Randomize question order", value=True)

st.title("Business Essentials — Chapters 1–6 Quiz")
st.markdown("Use this app to run quizzes built from the provided textbook and test bank. You can upload PDFs to re-run extraction (sidebar).")

# --- Quiz setup section ---
if not questions:
    st.warning("No questions found in data/questions_ch1_6.json. Upload files and run extraction or place the file in data/.")
    st.stop()

# Group by chapter and type
from collections import defaultdict
by_chapter = defaultdict(list)
by_type = defaultdict(list)
for q in questions:
    by_chapter[q["chapter"]].append(q)
    by_type[q["type"]].append(q)

# Build selection pool based on user preferences
def build_quiz_pool(total: int, mc_pref:int, tf_pref:int, sa_pref:int, ensure_one_chapter: bool) -> List[Dict]:
    pool = []
    # First: ensure at least one per chapter if requested
    if ensure_one_chapter:
        for ch in range(1,7):
            ch_list = by_chapter.get(ch, [])
            if ch_list:
                pool.append(random.choice(ch_list))
    # Next: try per-type quotas
    def draw_from_list(lst, n):
        if not lst:
            return []
        return random.sample(lst, min(n, len(lst)))
    pool += draw_from_list(by_type.get("multiple_choice", []), mc_pref)
    pool += draw_from_list(by_type.get("true_false", []), tf_pref)
    pool += draw_from_list(by_type.get("short_answer", []), sa_pref)
    # If total > len(pool), fill with mixed random questions
    if total == 0:
        # use all questions
        pool = questions.copy()
    else:
        # fill remainder with random questions across all if needed
        if len(pool) < total:
            remaining = [q for q in questions if q not in pool]
            take = min(total - len(pool), len(remaining))
            if take > 0:
                pool += random.sample(remaining, take)
        else:
            # trim to total
            pool = random.sample(pool, total)
    if randomize_by_chapter:
        random.shuffle(pool)
    # final de-dup preserve up to total
    seen = set()
    unique_pool = []
    for q in pool:
        key = (q["chapter"], q.get("question")[:80])
        if key not in seen:
            seen.add(key)
            unique_pool.append(q)
        if total and len(unique_pool) >= total:
            break
    return unique_pool

quiz_pool = build_quiz_pool(total_questions, per_type_mc, per_type_tf, per_type_sa, ensure_one_per_chapter)
st.sidebar.markdown(f"Quiz pool prepared: {len(quiz_pool)} questions")

# Initialize session state
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []  # list of dicts with question, given_answer, correct, timestamp
if "quiz_pool" not in st.session_state or st.session_state.get("last_pool_size") != len(quiz_pool):
    st.session_state.quiz_pool = quiz_pool
    st.session_state.last_pool_size = len(quiz_pool)
    st.session_state.quiz_index = 0
    st.session_state.responses = []

# --- Main quiz flow ---
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Question")
    if st.session_state.quiz_index < len(st.session_state.quiz_pool):
        q = st.session_state.quiz_pool[st.session_state.quiz_index]
        st.markdown(f"**Chapter {q['chapter']} • Type: {q['type']}**")
        st.write(q["question"])
        # UI by type
        user_answer = None
        if q["type"] == "multiple_choice":
            opts = q.get("options") or []
            # prepare options for display; if options like "A. text", show text
            display_opts = []
            for o in opts:
                # strip leading label like 'A. '
                m = re.match(r'^\s*([A-Z0-9]+)[\.\)]\s*(.+)

3) data/questions_ch1_6.json (sample)
Create folder data/ and save as data/questions_ch1_6.json
```json
[
  {
    "chapter": 1,
    "type": "multiple_choice",
    "question": "1) What is the primary goal of a business? A) Reduce competition B) Earn profits for owners C) Expand government control D) Limit consumer choice",
    "options": [
      "A. Reduce competition",
      "B. Earn profits for owners",
      "C. Expand government control",
      "D. Limit consumer choice"
    ],
    "answer": "B",
    "explanation": "In a capitalistic system, businesses exist to earn profits for owners."
  },
  {
    "chapter": 2,
    "type": "true_false",
    "question": "True or False: The Foreign Corrupt Practices Act makes it illegal for U.S. firms to bribe foreign government officials.",
    "options": [
      "True",
      "False"
    ],
    "answer": "True",
    "explanation": "The FCPA prohibits bribery of foreign officials by U.S. companies."
  },
  {
    "chapter": 3,
    "type": "short_answer",
    "question": "List three common sources of funding for new businesses.",
    "options": null,
    "answer": null,
    "explanation": "Examples: personal savings, bank loans, angel investors/venture capital."
  }
]
