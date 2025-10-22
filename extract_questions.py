# --- Classify question types / extract options ---
TF_RE = re.compile(r'\b(True|False|T/F|T or F)\b', re.IGNORECASE)
BLANK_RE = re.compile(r'_{3,}|_{1,}\bBlank\b|_____', re.IGNORECASE)
ANSWER_KEY_RE = re.compile(r'Answer[:\s]*([A-D]|[1-9][0-9]?|True|False|T|F)', re.IGNORECASE)

def extract_options_from_block(block_text: str) -> List[str]:
    """
    Extract option lines such as 'A. text' or 'A) text' or '1. text'.
    Returns list of strings, e.g. ['A. Option text', 'B. ...'].
    """
    opts = []
    # Single-line raw string regex (must not be broken across lines)
    option_pattern = re.compile(r'(?m)^\s*([A-D]|[1-9][0-9]?)\s*[\.\)]\s*(.+)

Why this fixes the error
- The original regex string was split across lines (introducing a newline inside the quotes) and thus Python saw an unterminated string literal.
- The corrected code uses a single-line raw string r'...' with parentheses balanced and no unterminated quotes.
- The pattern uses (?m) multiline flag so ^ matches option line starts; it captures labels like A, B, C, D or numeric labels like 1, 2, 10 and the option text.

Additional suggestions (based on your test-bank content)
- Your test bank prints answers as "Answer: E" or "Answer: TRUE" — ANSWER_KEY_RE above should capture those.
- Many questions include duplicate repeated pages or OCR artifacts; keep logging of maybe_unmatched questions (your script already writes logs/maybe_unmatched.json).
- Use extract_text_from_pdf to read .txt files if you have them instead of PDFs.

If you want, I can:
- Provide the full corrected extract_questions.py (I previously supplied a corrected full file — ensure you used that version).
- Run a quick lint or show a minimal unit test for extract_options_from_block using some sample question strings extracted from your PDF chunks (e.g., the lines you provided).
Tell me which you prefer and I'll supply the code/test., re.IGNORECASE)
    for m in option_pattern.finditer(block_text):
        label = m.group(1).strip()
        text = m.group(2).strip()
        opts.append(f"{label}. {text}")
    # Fallback: split by semicolons or pipes for inline options
    if not opts:
        parts = re.split(r'\s*;\s*|\s*\|\s*', block_text)
        candidate = [p.strip() for p in parts if 2 < len(p) < 200]
        if len(candidate) >= 2:
            opts = [f"{chr(65+i)}. {c}" for i, c in enumerate(candidate[:6])]
    return opts
