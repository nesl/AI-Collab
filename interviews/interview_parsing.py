#!/usr/bin/env python3
# collapse_interviews.py
#
# Input:  transcript file with multiple WEBVTT sessions (e.g., output.txt)
# Output: CSV with columns: Session, Speaker, Paragraph
# Consecutive lines by the same speaker are merged into one paragraph.
# Interviewer JULIAN is excluded.

import re
import pandas as pd
from pathlib import Path

def parse_session(text: str, session_id: int):
    """Parse a single session, collapse consecutive speaker lines."""
    # Remove timestamps and cue indices
    TIMESTAMP_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\b")
    text = TIMESTAMP_RE.sub("", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    blocks = re.split(r"\n\s*\n", text.strip())
    utterances = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        joined = " ".join(lines)

        # Match formats
        m = re.match(r"^\s*(?:>>\s*)?([A-Za-z][A-Za-z0-9 ()/_\-\.\']{0,60})\s*:\s*(.*)$", joined)
        if m:
            speaker, content = m.group(1).strip(), m.group(2).strip()
            if content:
                utterances.append((speaker, content))
            continue

        m2 = re.match(r"^\s*<v\s+([^>]+)>(.*?)</v>\s*$", joined)
        if m2:
            speaker, content = m2.group(1).strip(), m2.group(2).strip()
            if content:
                utterances.append((speaker, content))
            continue

        # Continuation of previous
        if utterances:
            prev_spk, prev_txt = utterances[-1]
            utterances[-1] = (prev_spk, (prev_txt + " " + joined).strip())

    # Collapse consecutive by same speaker
    collapsed = []
    for spk, txt in utterances:
        if collapsed and collapsed[-1][0] == spk:
            collapsed[-1] = (spk, collapsed[-1][1] + " " + txt)
        else:
            collapsed.append((spk, txt))

    # Filter out interviewer JULIAN
    rows = [
        {"Session": session_id, "Speaker": spk, "Paragraph": para}
        for spk, para in collapsed
    ]
    return rows

def process_file(input_path: str, output_csv: str):
    raw = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    # Split into sessions at WEBVTT markers
    session_splits = re.split(r"(?i)\bWEBVTT\b", raw)
    sessions = [s for s in session_splits if s.strip()]

    all_rows = []
    for i, session_text in enumerate(sessions, start=1):
        all_rows.extend(parse_session(session_text, i))

    df = pd.DataFrame(all_rows, columns=["Session", "Speaker", "Paragraph"])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    # Example usage
    process_file("output.txt", "interview_paragraphs_by_session.csv")
