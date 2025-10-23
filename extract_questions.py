#!/usr/bin/env python3
# extract_questions.py  –  works WITHOUT pdfplumber
# -----------------------------------------------------------
# 1. Extract text from textbook + test-bank PDF/TXT
# 2. Detect chapter ranges for ch. 1-6
# 3. Parse test-bank into question blocks
# 4. Map each block to a chapter (label first, else fuzzy)
# 5. Classify type (MCQ, T/F, short-answer, fill-blank)
# 6. Save clean JSON + logs
# -----------------------------------------------------------

from __future__ import annotations
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# PDF fallback: standard-lib-only pypdf (pre-installed on Streamlit Cloud)
try:
    import pypdf  # type: ignore
    HAS_PDF = True
except ModuleNotFoundError:  # local safety
    HAS_PDF = False

from rapidfuzz import fuzz
from tqdm import tqdm

# ---------- logging ----------
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

# ---------- text extraction ----------
def extract_text_from_pdf(path: str) -> str:
    """Return plain text from PDF (via pypdf) or TXT."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")

    if not HAS_PDF:
        raise RuntimeError("pypdf not available – cannot extract PDF text")

    text_parts: List[str] = []
    with p.open("rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n\n".join(text_parts)


# ---------- chapter detection ----------
CHAPTER_HDR_PATTERNS = [
    re.compile(r"^\s*Chapter\s+(\d{1,2})\b", re.I),
    re.compile(r"^\s*CHAPTER\s+(\d{1,2})\b", re.I),
    re.compile(r"^\s*Ch\.\s*(\d{1,2})\b", re.I),
    re.compile(r"^\s*(\d{1,2})\s+The\s+", re.I),
]

def detect_chapters(book_text: str, chapters_to_find: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> Dict[int, Dict]:
    """Return {chap_num: {'start':i,'end':i,'title':str,'text':str}}."""
    idxs: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"(?m)^.*$", book_text):
        line, start = m.group(0), m.start()
        for pat in CHAPTER_HDR_PATTERNS:
            hit = pat.search(line)
            if hit:
                num = int(hit.group(1))
                if num in chapters_to_find:
                    idxs.append((num, start, line.strip()))
                break

    # fuzzy fallback for missing chapters
    found_nums = {t[0] for t in idxs}
    for num in chapters_to_find:
        if num in found_nums:
            continue
        target = f"chapter {num}"
        best_score, best_pos = 0, None
        sample = book_text[:500_000]
        for m in re.finditer(r"(?i)chapter[\s\._-]*\w{1,10}", sample):
            score = fuzz.partial_ratio(target, m.group(0))
            if score > best_score:
                best_score, best_pos = score, m.start()
        if best_score >= 80 and best_pos is not None:
            idxs.append((num, best_pos, f"fuzzy:{target} (score={best_score})"))
            logging.info("Fuzzy-located chapter %d at pos %d (score=%d)", num, best_pos, best_score)

    idxs_sorted = sorted(idxs, key=lambda t: t[1])
    chapter_map: Dict[int, Dict] = {}
    for i, (num, start, heading) in enumerate(idxs_sorted):
        end = idxs_sorted[i + 1][1] if i + 1 < len(idxs_sorted) else len(book_text)
        chapter_map[num] = {"start": start, "end": end, "title": heading, "text": book_text[start:end].strip()}
        logging.info("Chapter %d mapped start=%d end=%d title=%s", num, start, end, heading)
    return chapter_map


# ---------- test-bank parsing ----------
def parse_testbank_text(tb_text: str) -> List[Dict[str, str]]:
    """Split into question blocks; return [{'raw':..., 'short':...}, ...]."""
    lines = tb_text.splitlines()
    blocks: List[Dict[str, str]] = []
    current: List[str] = []
    starter = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d+|(\d{1,3})[.\)])", re.I)
    for line in lines:
        if starter.match(line):
            if current:
                blocks.append({"raw": "\n".join(current).strip()})
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        blocks.append({"raw": "\n".join(current).strip()})
    # fallback: blank-line split
    if not blocks:
        for part in re.split(r"\n\s*\n", tb_text):
            part = part.strip()
            if part:
                blocks.append({"raw": part})
    for b in blocks:
        b["short"] = b["raw"].splitlines()[0][:200]
    logging.info("Parsed %d question blocks from test bank", len(blocks))
    return blocks


# ---------- chapter label inside question ----------
CH_LABEL_RE = re.compile(r"chapter\s*(\d{1,2})", re.I)

def extract_chapter_label(block: str) -> Optional[int]:
    m = CH_LABEL_RE.search(block)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


# ---------- fuzzy mapping ----------
def map_question_to_chapter(block: str, chapter_texts: Dict[int, Dict], threshold: int = 70) -> Optional[int]:
    """Return chapter number or None."""
    # 1. explicit label
    lab = extract_chapter_label(block)
    if lab and lab in chapter_texts:
        return lab
    # 2. fuzzy match
    best_score, best_ch = 0, None
    snippet = " ".join(block.split())[:1000]
    for ch_num, info in chapter_texts.items():
        score_title = fuzz.partial_ratio(snippet, info["title"])
        score_body = fuzz.partial_ratio(snippet, info["text"][:2000])
        score = max(score_title, score_body)
        if score > best_score:
            best_score, best_ch = score, ch_num
    if best_score >= threshold:
        return best_ch
    logging.debug("Low confidence (score=%d) for block: %s", best_score, block[:120])
    return None


# ---------- question classification ----------
TF_RE = re.compile(r"\b(True|False|T/F|T or F)\b", re.I)
BLANK_RE = re.compile(r"_{3,}|_{1,}\bBlank\b|_____", re.I)
ANSWER_RE = re.compile(r"Answer[:\s]*([A-D]|[1-9][0-9]?|True|False|T|F)\b", re.I)
OPTION_RE = re.compile(r"(?m)^\s*([A-D]|[1-9][0-9]?)\s*[\.\)]\s*(.+)$", re.I)

def extract_options(block: str) -> List[str]:
    """Return ['A. text', 'B. text', ...] or []."""
    opts = []
    for m in OPTION_RE.finditer(block):
        label, text = m.group(1), m.group(2).strip()
        opts.append(f"{label}. {text}")
    if not opts:
        # inline fallback: semicolon / pipe split
        parts = re.split(r"\s*;\s*|\s*\|\s*", block)
        cand = [p.strip() for p in parts if 2 < len(p) < 200]
        if len(cand) >= 2:
            opts = [f"{chr(65+i)}. {c}" for i, c in enumerate(cand[:6])]
    return opts

def classify_question(block: str) -> Tuple[str, List[str], Optional[str]]:
    """Return (type, options, answer_key)."""
    opts = extract_options(block)
    ans_match = ANSWER_RE.search(block)
    answer = ans_match.group(1).strip() if ans_match else None

    if TF_RE.search(block) and (any("True" in o or "False" in o for o in opts) or re.search(r"\bTrue\b|\bFalse\b", block, re.I)):
        return "true_false", opts or ["True", "False"], answer
    if BLANK_RE.search(block):
        return "fill_in_blank", [], answer
    if opts:
        return "multiple_choice", opts, answer
    if block.strip().endswith("?") or len(block.split()) < 40:
        return "short_answer", [], answer
    return "unknown", [], answer


# ---------- build final dataset ----------
def build_questions_dataset(
    book_path: str,
    testbank_path: str,
    out_json_path: str = "data/questions_ch1_6.json",
    chapters: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    fuzzy_threshold: int = 70,
) -> Dict:
    """Full pipeline."""
    book_text = extract_text_from_pdf(book_path)
    ch_map = detect_chapters(book_text, chapters_to_find=chapters)

    tb_text = extract_text_from_pdf(testbank_path)
    blocks = parse_testbank_text(tb_text)

    out_questions: List[Dict] = []
    maybe_unmatched: List[Dict] = []
    matched_count = 0

    for blk in tqdm(blocks, desc="Mapping/classifying"):
        raw = blk["raw"]
        mapped = map_question_to_chapter(raw, ch_map, threshold=fuzzy_threshold)
        if mapped and mapped in chapters:
            qtype, opts, ans = classify_question(raw)
            out_questions.append(
                {
                    "chapter": mapped,
                    "type": qtype,
                    "question": raw,
                    "options": opts or None,
                    "answer": ans,
                    "explanation": None,
                }
            )
            matched_count += 1
        else:
            maybe_unmatched.append({"raw": raw, "mapped": mapped})

    # write outputs
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
        "questions_found": len(blocks),
        "matched_to_ch1_6": matched_count,
        "maybe_unmatched_count": len(maybe_unmatched),
        "out_json": str(out_path),
        "maybe_unmatched_json": str(amb_path),
        "fuzzy_threshold": fuzzy_threshold,
    }
    logging.info("Extraction complete: %s", summary)
    return summary


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract & map test-bank questions to chapters 1-6")
    parser.add_argument("--book", required=True, help="textbook PDF/TXT")
    parser.add_argument("--testbank", required=True, help="test-bank PDF/TXT")
    parser.add_argument("--out", default="data/questions_ch1_6.json", help="output JSON")
    parser.add_argument("--threshold", type=int, default=70, help="fuzzy match threshold 0-100")
    args = parser.parse_args()

    summary = build_questions_dataset(
        args.book,
        args.testbank,
        out_json_path=args.out,
        fuzzy_threshold=args.threshold,
    )
    print("Summary:", summary)
