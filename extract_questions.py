"""
extract_questions.py

Helpers for extracting book and test-bank text, detecting chapter ranges (1-6),
mapping questions to chapters (direct label OR fuzzy match), classifying question types,
and exporting filtered questions to JSON. Logs ambiguous / maybe-unmatched items.

Unit-testable functions:
- extract_text_from_pdf(path) -> str
- detect_chapters(book_text) -> dict
- parse_testbank_text(tb_text) -> List[dict]
- map_question_to_chapter(question_text, chapter_texts, threshold=70) -> Optional[int]
- classify_question(block_text) -> (type, options, answer)
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    # If user provided a .txt file, just read it
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
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
    r"^\s*Chapter\s+(One|Two|Three|Four|Five|Six)\b",  # Chapter One etc.
    r"^\s*(\d{1,2})\s+The\s+",               # "1 The ..." (sometimes)
]

def detect_chapters(book_text: str, chapters_to_find: Tuple[int, ...] = (1,2,3,4,5,6)) -> Dict[int, Dict]:
    """
    Detect approximate text spans for requested chapters.
    Returns dict mapping chapter_number -> {'start': idx, 'end': idx, 'title': str, 'text': str}
    """
    chapters_to_find = tuple(chapters_to_find)
    idxs = []  # list of tuples (chapter_number, start_char_index, heading_line)
    # iterate through text lines to find headings with char index
    for match in re.finditer(r"(?m)^.*$", book_text):
        line = match.group(0)
        start = match.start()
        for pat in CHAPTER_HDR_PATTERNS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                # try to extract number
                num = None
                if m.groups():
                    g = m.group(1)
                    try:
                        num = int(g)
                    except Exception:
                        WORD_NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}
                        num = WORD_NUM.get(str(g).lower())
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
        text_sample = book_text[:500000]
        for m in re.finditer(r"(?i)chapter[\s\._-]*\w{1,10}", text_sample):
            cand = m.group(0)
            score = fuzz.partial_ratio(target, cand)
            if score > best_score:
                best_score = score
                best_pos = m.start()
        if best_score >= 80 and best_pos is not None:
            idxs.append((n, best_pos, f"fuzzy:{target} (score={best_score})"))
            logging.info("Fuzzy-located chapter %d at pos %d (score=%d)", n, best_pos, best_score)

    idxs_sorted = sorted(idxs, key=lambda x: x[1])
    chapter_map: Dict[int, Dict] = {}
    for i, (num, start, heading) in enumerate(idxs_sorted):
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
    Heuristic: detect starters (Q1, Question 1, 1)  and split blocks.
    """
    lines = tb_text.splitlines()
    q_blocks = []
    current_lines = []
    starter_re = re.compile(r'^\s*(?:Q(?:uestion)?\s*\d+|(\d{1,3})[.)])\s*', re.IGNORECASE)
    for line in lines:
        if starter_re.match(line):
            if current_lines:
                q_blocks.append({"raw": "\n".join(current_lines).strip()})
            current_lines = [line]
        else:
            if current_lines:
                current_lines.append(line)
            else:
                # stray text before first question: ignore or accumulate
                continue
    if current_lines:
        q_blocks.append({"raw": "\n".join(current_lines).strip()})
    if not q_blocks:
        parts = re.split(r'\n\s*\n', tb_text)
        for p in parts:
            p = p.strip()
            if p:
                q_blocks.append({"raw": p})
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
    Map a question to a chapter using direct label or rapidfuzz partial_ratio.
    """
    lab = extract_chapter_label_from_block(question_text)
    if lab and lab in chapter_texts:
        logging.debug("Direct label found chapter %d in question", lab)
        return lab

    best_score = 0
    best_ch = None
    question_snippet = " ".join(question_text.split())[:1000]
    for ch_num, ch_info in chapter_texts.items():
        title = ch_info.get("title", "")
        score_title = fuzz.partial_ratio(question_snippet, title)
        chapter_snippet = ch_info.get("text", "")[:2000]
        score_body = fuzz.partial_ratio(question_snippet, chapter_snippet)
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
TF_RE = re.compile(r'\b(True|False|T/F|T or F)\b', re.IGNORECASE)
BLANK_RE = re.compile(r'_{3,}|_{1,}\bBlank\b|_____', re.IGNORECASE)
ANSWER_KEY_RE = re.compile(r'Answer[:\s]*([A-D]|[1-9][0-9]?|True|False|T|F)', re.IGNORECASE)

def extract_options_from_block(block_text: str) -> List[str]:
    """
    Extract option lines such as 'A. text' or 'A) text' or '1. text'.
    Returns list of strings.
    """
    opts = []
    # Correctly-formed regex (single-line string) to avoid unterminated string issues
    pattern = re.compile(r'(?m)^\s*([A-D]|[1-9][0-9]?)\s*[\.\)]\s*(.+)

Why this fixes the SyntaxError
- The original error came from a regex literal that was split across lines (unterminated string). In the fixed file the regex strings are single-line strings or properly quoted, avoiding unterminated string literals.
- I also added a small guard for reading .txt files and improved some regex robustness.

Next steps / suggestions
- Replace your extract_questions.py with the corrected file above.
- Re-run: python extract_questions.py --book Business_Essentials_13th_Ed.pdf --testbank Stuvia-7183014-test-bank-for-business-essentials13th-edition-by-ronald-j.-ebert-ricky-w.-griffin-chapter-1-17.pdf --out data/questions_ch1_6.json --threshold 70
- If another error appears, paste the full traceback and I’ll help iterate quickly.
- If you want, I can also provide a single-file combined script (extract + app) or add more unit tests.

If you prefer, I can show a minimal diff (just the corrected regex) instead of replacing the whole file., re.IGNORECASE)
    for m in pattern.finditer(block_text):
        label = m.group(1).strip()
        text = m.group(2).strip()
        opts.append(f"{label}. {text}")
    # fallback: split by semicolons if no labeled options found
    if not opts:
        parts = re.split(r'\s*;\s*|\s*\|\s*', block_text)
        candidate = [p.strip() for p in parts if 2 < len(p) < 200]
        if len(candidate) >= 2:
            opts = [f"{chr(65+i)}. {c}" for i,c in enumerate(candidate[:6])]
    return opts

def classify_question(block_text: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Classify question type and attempt to extract options and answer.
    """
    opts = extract_options_from_block(block_text)
    answer = None
    m = ANSWER_KEY_RE.search(block_text)
    if m:
        answer = m.group(1).strip()
    if TF_RE.search(block_text) and (any(re.search(r'\bTrue\b|\bFalse\b', o, re.IGNORECASE) for o in opts) or re.search(r'\bTrue\b|\bFalse\b', block_text, re.IGNORECASE)):
        return "true_false", opts or ["True", "False"], answer
    if BLANK_RE.search(block_text):
        return "fill_in_blank", [], answer
    if opts:
        return "multiple_choice", opts, answer
    if block_text.strip().endswith('?') or len(block_text.split()) < 40:
        return "short_answer", [], answer
    return "unknown", [], answer

# --- Compose final dataset and write JSON ---
def build_questions_dataset(book_path: str,
                            testbank_path: str,
                            out_json_path: str = "data/questions_ch1_6.json",
                            chapters=(1,2,3,4,5,6),
                            fuzzy_threshold: int = 70,
                            include_maybe_unmatched: bool = False) -> Dict:
    """
    Extracts text, detects chapters, parses test bank, maps and classifies questions, and writes JSON.
    Returns summary dict.
    """
    book_text = extract_text_from_pdf(book_path)
    logging.info("Extracted book text (%d chars)", len(book_text))
    ch_map = detect_chapters(book_text, chapters_to_find=chapters)
    for n in chapters:
        if n not in ch_map:
            logging.warning("Chapter %d not detected in book text", n)
    test_text = extract_text_from_pdf(testbank_path)
    logging.info("Extracted test bank text (%d chars)", len(test_text))
    q_blocks = parse_testbank_text(test_text)
    out_questions = []
    maybe_unmatched = []
    matched_count = 0
    for qb in tqdm(q_blocks, desc="Mapping questions"):
        raw = qb["raw"]
        explicit = extract_chapter_label_from_block(raw)
        mapped = None
        if explicit and explicit in ch_map:
            mapped = explicit
        else:
            mapped = map_question_to_chapter(raw, ch_map, threshold=fuzzy_threshold)
        if mapped and mapped in chapters:
            qtype, opts, ans = classify_question(raw)
            entry = {
                "chapter": int(mapped),
                "type": qtype,
                "question": raw,
                "options": opts if opts else None,
                "answer": ans,
                "explanation": None,
            }
            out_questions.append(entry)
            matched_count += 1
        else:
            maybe_unmatched.append({"raw": raw, "mapped": mapped})
            logging.debug("Maybe unmatched question (mapped=%s): %s", mapped, raw[:120])
    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_questions, f, indent=2, ensure_ascii=False)
    amb_path = LOG_DIR / "maybe_unmatched.json"
    with open(amb_path, "w", encoding="utf-8") as f:
        json.dump(maybe_unmatched, f, indent=2, ensure_ascii=False)
    summary = {
        "book_path": str(book_path),
        "testbank_path": str(testbank_path),
        "questions_found": len(q_blocks),
        "matched_to_ch1_6": matched_count,
        "maybe_unmatched_count": len(maybe_unmatched),
        "out_json": str(out_path),
        "maybe_unmatched_json": str(amb_path),
        "fuzzy_threshold": fuzzy_threshold
    }
    logging.info("Extraction complete: %s", summary)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and map questions from book and testbank.")
    parser.add_argument("--book", required=True, help="Path to textbook PDF or text")
    parser.add_argument("--testbank", required=True, help="Path to test bank PDF or text")
    parser.add_argument("--out", default="data/questions_ch1_6.json", help="Output JSON path")
    parser.add_argument("--threshold", type=int, default=70, help="Fuzzy match threshold (0-100)")
    args = parser.parse_args()
    summary = build_questions_dataset(args.book, args.testbank, args.out, fuzzy_threshold=args.threshold)
    print("Summary:", summary)
