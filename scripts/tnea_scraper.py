#!/usr/bin/env python3
"""
TNEA Cutoff Mark Scraper  v4
=============================
Downloads official TNEA General-Academic & Vocational provisional allotment
PDFs from static.tneaonline.org, then computes cutoff marks as:

    cutoff[college][branch][community] = MIN(AGGR MARK) across all allotted rows

OUTPUT FORMAT
-------------
[
  {
    "college_name":     "University Departments of Anna University Chennai - CEG Campus ...",
    "department":       "COMPUTER SCIENCE AND ENGINEERING",
    "counselling_code": "1",
    "cutoffs": { "OC": "200", "BC": "199.5", "MBC": "199.5", "SC": "197.5", "ST": "191.5" },
    "year": 2024
  }, ...
]

USAGE
-----
  pip install pdfplumber requests
  python3 tnea_scraper.py                     # all years (2023 2024 2025)
  python3 tnea_scraper.py --years 2024        # single year
  python3 tnea_scraper.py --codes 1 2 3       # filter colleges
  python3 tnea_scraper.py --out results.json  # custom output
  python3 tnea_scraper.py --pdf-dir ./pdfs/   # custom cache dir
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

for _pkg in ("pdfplumber", "requests"):
    try:
        __import__(_pkg)
    except ImportError:
        os.system(f"pip install {_pkg} --break-system-packages -q")

import pdfplumber
import requests

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIRMED WORKING PDF URLs  (verified by direct fetch / search snippets)
# ─────────────────────────────────────────────────────────────────────────────
PDF_SOURCES = [
    # ── 2023 ──────────────────────────────────────────────────────────────
    (2023, "gen_r1",
     "https://static.tneaonline.org/docs/General-Academic-Round1-allotment.pdf"
     "?t=1691798400051"),
    (2023, "gen_r2",
     "https://static.tneaonline.org/docs/GENERAL-ACADEMIC-PROVISIONAL-ALLOTMENT-LIST-round2.pdf"
     "?t=1693094400051"),
    (2023, "gen_r3",
     "https://static.tneaonline.org/docs/ROUND-3-GENERAL-ACADEMIC-PROVISIONAL-ALLOTMENT-LIST.pdf"
     "?t=1694563200023"),
    (2023, "gen_sup",
     "https://static.tneaonline.org/docs/SUPPLEMENTARY-GENERAL-ACADEMIC-ALLOTMENT-LIST.pdf"
     "?t=1695254400022"),
    (2023, "voc",
     "https://static.tneaonline.org/docs/Vocational_2023_Mark_Cutoff.pdf"
     "?t=1742256000040"),

    # ── 2024 ──────────────────────────────────────────────────────────────
    (2024, "gen_r1",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND1_2024.pdf"
     "?t=1741392000023"),
    (2024, "gen_r2",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND2_2024.pdf"
     "?t=1741478400023"),
    (2024, "gen_r3",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND3_2024.pdf"
     "?t=1741478400023"),
    (2024, "gen_sup",
     "https://static.tneaonline.org/docs/SUPPLEMENTARY-GENERAL-ACADEMIC-ALLOTMENT-LIST-2024.pdf"
     "?t=1726358400022"),
    (2024, "voc",
     "https://static.tneaonline.org/docs/Vocational_2024_Mark_Cutoff.pdf"
     "?t=1742256000040"),

    # ── 2025 (probed; will be skipped gracefully if not yet published) ─────
    (2025, "gen_r1",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND1_2025.pdf"
     "?t=1753920000023"),
    (2025, "gen_r2",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND2_2025.pdf"
     "?t=1754006400023"),
    (2025, "gen_r3",
     "https://static.tneaonline.org/docs/GENERAL_ACADEMIC_ROUND3_2025.pdf"
     "?t=1754352000023"),
    (2025, "gen_sup",
     "https://static.tneaonline.org/docs/SUPPLEMENTARY-GENERAL-ACADEMIC-ALLOTMENT-LIST-2025.pdf"
     "?t=1756944000022"),
    (2025, "voc",
     "https://static.tneaonline.org/docs/Vocational_2025_Mark_Cutoff.pdf"
     "?t=1756944000022"),
]

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES      = ["OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"]
COMM_VALID      = {"OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"}
COMM_NORMALISE  = {"MBCV": "MBC", "MBCDNC": "MBC", "BCM": "BCM"}

NUM_RE   = re.compile(r"^\d{1,3}(\.\d+)?$")
CODE_RE  = re.compile(r"^\d{1,4}$")
BCODE_RE = re.compile(r"^[A-Z]{1,3}\d{0,2}$")
DATE_RE  = re.compile(r"^\d{2}-\d{2}-\d{4}$")

SKIP_RE = re.compile(
    r"(^\s*S\.?\s*NO|APPLN|APPLICATION|COLLEGE\s*CODE|BRANCH\s*CODE|"
    r"ALLOTTED\s*CAT|DIRECTORATE|TAMILNADU|PROVISIONAL|ROUND\s*[0-9]|"
    r"SUPPLEMENTARY|VOCATIONAL|MARK\s*CUTOFF|^COMMUNITY$|^AGGREGATE$|"
    r"COMMUN|GENERAL\s*RANK|GOVT\s*RANK)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Allotment PDF column layouts (by number of columns after table extraction)
#
#  Layout A (10 cols) — Standard General Academic:
#    0:S_NO  1:APPLN  2:NAME  3:DOB  4:COMMUNITY  5:AGGR_MARK  6:RANK
#    7:COLLEGE_CODE   8:BRANCH_CODE   9:ALLOTTED_CAT
#
#  Layout B (9 cols) — Supplementary (no DOB):
#    0:S_NO  1:APPLN  2:NAME  3:COMMUNITY  4:AGGR_MARK  5:RANK
#    6:COLLEGE_CODE   7:BRANCH   8:ALLOTTED_CAT
#
#  Layout C (8 cols) — Old supplementary variant
#    0:S_NO  1:APPLN  2:NAME  3:COMMUNITY  4:AGGR_MARK  5:COLLEGE_CODE
#    6:BRANCH  7:ALLOTTED_CAT
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_community(s: str) -> str | None:
    s = s.strip().upper()
    s = COMM_NORMALISE.get(s, s)
    return s if s in COMM_VALID else None


def _parse_allotment_row_positional(cells: list[str]) -> dict | None:
    """Parse one row using positional column logic."""
    n = len(cells)

    # ── Layout A: 10 columns ──────────────────────────────────────────────
    if n >= 10:
        comm     = _normalise_community(cells[4])
        mark_raw = cells[5].strip()
        ccode    = cells[7].strip()
        bcode    = cells[8].strip()
        if comm and NUM_RE.match(mark_raw) and CODE_RE.match(ccode) and BCODE_RE.match(bcode):
            return {"community": comm, "mark": float(mark_raw),
                    "college_code": ccode, "branch_code": bcode}

    # ── Layout B: 9 columns ───────────────────────────────────────────────
    if n == 9:
        comm     = _normalise_community(cells[3])
        mark_raw = cells[4].strip()
        ccode    = cells[6].strip()
        bcode    = cells[7].strip()
        if comm and NUM_RE.match(mark_raw) and CODE_RE.match(ccode) and BCODE_RE.match(bcode):
            return {"community": comm, "mark": float(mark_raw),
                    "college_code": ccode, "branch_code": bcode}

    # ── Layout C: 8 columns ───────────────────────────────────────────────
    if n == 8:
        comm     = _normalise_community(cells[3])
        mark_raw = cells[4].strip()
        ccode    = cells[5].strip()
        bcode    = cells[6].strip()
        if comm and NUM_RE.match(mark_raw) and CODE_RE.match(ccode) and BCODE_RE.match(bcode):
            return {"community": comm, "mark": float(mark_raw),
                    "college_code": ccode, "branch_code": bcode}

    return None


def _parse_allotment_row_heuristic(tokens: list[str]) -> dict | None:
    """
    Fallback: scan tokens by type.
    Key insight: COMMUNITY always appears before AGGR_MARK,
                 and COLLEGE_CODE appears after RANK.
    We locate community first, then take the next numeric as AGGR_MARK,
    then skip RANK, then read COLLEGE_CODE and BRANCH_CODE.
    """
    tokens = [t.strip() for t in tokens if t.strip()]
    if len(tokens) < 5:
        return None

    # Find community position
    comm_idx = None
    comm_val = None
    for i, t in enumerate(tokens):
        c = _normalise_community(t)
        if c:
            comm_idx = i
            comm_val = c
            break
    if comm_idx is None:
        return None

    # After community: next float ≤ 200 is AGGR_MARK
    mark = None
    mark_idx = None
    for i in range(comm_idx + 1, len(tokens)):
        t = tokens[i]
        if NUM_RE.match(t):
            v = float(t)
            if v <= 200:
                mark = v
                mark_idx = i
                break
    if mark is None:
        return None

    # After mark: next number is RANK (skip it), then COLLEGE_CODE, then BRANCH_CODE
    numeric_after_mark = []
    branch_candidates = []
    for i in range(mark_idx + 1, len(tokens)):
        t = tokens[i]
        if CODE_RE.match(t):
            numeric_after_mark.append((i, t))
        elif BCODE_RE.match(t):
            branch_candidates.append((i, t))

    # college_code = first CODE_RE after mark that is NOT immediately followed by a RANK
    # branch_code  = first BCODE_RE after the college_code
    ccode = None
    bcode = None

    if len(numeric_after_mark) >= 2:
        # skip first (it's the rank), take second as college code
        ccode = numeric_after_mark[1][1]
        ccode_idx = numeric_after_mark[1][0]
    elif len(numeric_after_mark) == 1:
        ccode = numeric_after_mark[0][1]
        ccode_idx = numeric_after_mark[0][0]
    else:
        return None

    # branch code: first BCODE_RE after college code index
    for idx, t in branch_candidates:
        if idx > ccode_idx:
            bcode = t
            break

    if ccode and bcode:
        return {"community": comm_val, "mark": mark,
                "college_code": ccode, "branch_code": bcode}
    return None


def parse_allotment_pdf(pdf_path: Path) -> list[dict]:
    """Parse a TNEA allotment-list PDF; return list of row dicts."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Strategy 1: table mode (clean columns)
            page_rows: list[list[str]] = []
            for settings in [
                {"vertical_strategy": "lines",       "horizontal_strategy": "lines"},
                {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
            ]:
                tables = page.extract_tables(settings) or []
                if tables:
                    for tbl in tables:
                        for row in tbl:
                            page_rows.append([str(c or "").strip() for c in row])
                    break

            if page_rows:
                for cells in page_rows:
                    if SKIP_RE.search(" ".join(cells)):
                        continue
                    rec = _parse_allotment_row_positional(cells)
                    if rec:
                        rows.append(rec)
                continue

            # Strategy 2: text + heuristic
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line or SKIP_RE.search(line):
                    continue
                tokens = re.split(r"\s{2,}", line)
                rec = _parse_allotment_row_heuristic(tokens)
                if rec:
                    rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Vocational cutoff PDF parser  (pre-compiled table)
# ─────────────────────────────────────────────────────────────────────────────

def parse_vocational_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse Vocational_YYYY_Mark_Cutoff.pdf.
    Column layout: COLLEGE CODE | COLLEGE NAME | BRANCH CODE | BRANCH NAME | OC | BC | BCM | MBC | SC | SCA | ST
    """
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try table extraction first
            for settings in [
                {"vertical_strategy": "lines",       "horizontal_strategy": "lines"},
                {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
            ]:
                tables = page.extract_tables(settings) or []
                if tables:
                    for tbl in tables:
                        rows = [[str(c or "").strip() for c in row] for row in tbl if row]
                        records.extend(_parse_voc_table_rows(rows))
                    break
            else:
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                records.extend(_parse_voc_text(text))
    return records


def _parse_voc_table_rows(rows: list[list[str]]) -> list[dict]:
    out = []
    for row in rows:
        if len(row) < 5:
            continue
        joined = " ".join(row)
        if SKIP_RE.search(joined):
            continue
        ccode = row[0].strip()
        if not CODE_RE.match(ccode):
            continue
        # Find branch code
        br_idx = None
        for i in range(1, min(5, len(row))):
            if BCODE_RE.match(row[i].strip()):
                br_idx = i
                break
        if br_idx is None:
            continue
        cname  = " ".join(row[1:br_idx]).strip()
        bcode  = row[br_idx].strip()
        bname  = row[br_idx + 1].strip() if br_idx + 1 < len(row) else ""
        cuts   = row[br_idx + 2:] if br_idx + 2 < len(row) else []
        cutoffs = {}
        for cat, v in zip(CATEGORIES, cuts[:7]):
            v = v.strip()
            if re.match(r"^\d{2,3}(\.\d+)?$", v):
                cutoffs[cat] = v
        if cutoffs:
            out.append({"college_code": ccode, "branch_code": bcode,
                        "college_name": cname, "branch_name": bname.upper(),
                        "cutoffs": cutoffs})
    return out


def _parse_voc_text(text: str) -> list[dict]:
    """Text-mode fallback for vocational PDFs."""
    records = []
    college_code = None
    college_name_parts: list[str] = []
    branch_code  = None
    branch_name_parts: list[str] = []
    cut_vals: list[str] = []

    def flush():
        nonlocal branch_code, branch_name_parts, cut_vals
        if college_code and branch_code and cut_vals:
            cutoffs = {}
            for cat, v in zip(CATEGORIES, cut_vals):
                v = v.strip()
                if re.match(r"^\d{2,3}(\.\d+)?$", v):
                    cutoffs[cat] = v
            if cutoffs:
                records.append({
                    "college_code": college_code,
                    "branch_code":  branch_code,
                    "college_name": " ".join(college_name_parts).strip(),
                    "branch_name":  " ".join(branch_name_parts).strip().upper(),
                    "cutoffs":      cutoffs,
                })
        branch_code = None
        branch_name_parts.clear()
        cut_vals.clear()

    for line in [l.strip() for l in text.split("\n") if l.strip()]:
        if SKIP_RE.search(line):
            continue
        tokens = re.split(r"\s{2,}", line)
        if tokens and CODE_RE.match(tokens[0]):
            flush()
            college_code = tokens[0]
            college_name_parts = [" ".join(tokens[1:])]
        elif tokens and BCODE_RE.match(tokens[0]) and college_code:
            flush()
            branch_code = tokens[0]
            rest = tokens[1:]
            j = next((i for i, v in enumerate(rest)
                       if re.match(r"^\d{2,3}(\.\d+)?$", v)), len(rest))
            branch_name_parts = rest[:j]
            cut_vals = rest[j:]
        elif college_code and not branch_code:
            college_name_parts.append(line)
        elif branch_code:
            if not cut_vals:
                for i, v in enumerate(tokens):
                    if re.match(r"^\d{2,3}(\.\d+)?$", v):
                        branch_name_parts.extend(tokens[:i])
                        cut_vals = tokens[i:]
                        break
                else:
                    branch_name_parts.append(line)
            else:
                cut_vals.extend(tokens)
    flush()
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Lookup tables
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_COLLEGE_NAMES = {
    "1":    "University Departments of Anna University Chennai - CEG Campus Sardar Patel Road Guindy Chennai 600 025",
    "2":    "University Departments of Anna University Chennai - ACT Campus Sardar Patel Road Guindy Chennai 600 025",
    "3":    "University Departments of Anna University Chennai - SAP Campus Sardar Patel Road Guindy Chennai 600 025",
    "1004": "Anna University MIT Campus Chennai 600 044",
    "3011": "University College of Engineering Tiruchirappalli (BIT Campus) Tiruchirappalli 620 024",
    "3015": "University College of Engineering Villupuram",
    "3016": "University College of Engineering Arni",
    "3017": "University College of Engineering Tindivanam",
    "3018": "University College of Engineering Ramanathapuram",
    "4020": "Anna University Regional Campus Tirunelveli",
    "4023": "University College of Engineering Nagercoil",
    "4026": "University College of Engineering Kancheepuram",
}

BUILTIN_BRANCH_NAMES = {
    "AD":"ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
    "AE":"AERONAUTICAL ENGINEERING",
    "AG":"AGRICULTURAL ENGINEERING",
    "AL":"ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING",
    "AM":"CSE (AI AND MACHINE LEARNING)",
    "AO":"AEROSPACE ENGINEERING",
    "AP":"APPAREL TECHNOLOGY",
    "AT":"ARTIFICIAL INTELLIGENCE AND DATA SCIENCE (SS)",
    "AU":"AUTOMOBILE ENGINEERING",
    "BM":"BIO MEDICAL ENGINEERING",
    "BT":"BIO TECHNOLOGY",
    "BY":"BIO MEDICAL ENGINEERING (SS)",
    "CB":"COMPUTER SCIENCE AND BUSINESS SYSTEMS",
    "CD":"COMPUTER SCIENCE AND DESIGN",
    "CE":"CIVIL ENGINEERING",
    "CF":"COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE)",
    "CG":"COMPUTER SCIENCE AND ENGINEERING (AI AND MACHINE LEARNING) (SS)",
    "CH":"CHEMICAL ENGINEERING",
    "CI":"COMPUTER SCIENCE AND ENGINEERING (INTERNET OF THINGS)",
    "CJ":"COMPUTER SCIENCE AND ENGINEERING (5 YEARS)",
    "CK":"CIVIL ENGINEERING (ENVIRONMENTAL ENGINEERING)",
    "CL":"CHEMICAL ENGINEERING (SS)",
    "CM":"COMPUTER SCIENCE AND ENGINEERING (SS)",
    "CN":"CIVIL ENGINEERING (SS)",
    "CO":"COMPUTER AND COMMUNICATION ENGINEERING",
    "CR":"CERAMIC TECHNOLOGY (SS)",
    "CS":"COMPUTER SCIENCE AND ENGINEERING",
    "CW":"COMPUTER SCIENCE AND BUSINESS SYSTEMS (SS)",
    "CY":"CSE (CYBER SECURITY)",
    "EC":"ELECTRONICS AND COMMUNICATION ENGINEERING",
    "EE":"ELECTRICAL AND ELECTRONICS ENGINEERING",
    "EF":"ELECTRICAL AND COMPUTER ENGINEERING",
    "EI":"ELECTRONICS AND INSTRUMENTATION ENGINEERING",
    "EL":"ELECTRONICS ENGINEERING (VLSI DESIGN AND TECHNOLOGY) (SS)",
    "EM":"ELECTRONICS AND COMMUNICATION ENGINEERING (SS)",
    "ES":"ELECTRICAL AND ELECTRONICS (SANDWICH) (SS)",
    "EV":"ELECTRONICS ENGINEERING (VLSI DESIGN AND TECHNOLOGY)",
    "EY":"ELECTRICAL AND ELECTRONICS ENGINEERING (SS)",
    "FD":"FOOD TECHNOLOGY",
    "FS":"FOOD TECHNOLOGY (SS)",
    "FT":"FASHION TECHNOLOGY",
    "FY":"FASHION TECHNOLOGY (SS)",
    "GI":"GEO INFORMATICS",
    "IB":"INDUSTRIAL BIO TECHNOLOGY",
    "IC":"INSTRUMENTATION AND CONTROL ENGINEERING",
    "ID":"INTERIOR DESIGN (SS)",
    "IM":"INFORMATION TECHNOLOGY (SS)",
    "IS":"INDUSTRIAL BIO TECHNOLOGY (SS)",
    "IT":"INFORMATION TECHNOLOGY",
    "IY":"INSTRUMENTATION AND CONTROL ENGINEERING (SS)",
    "LE":"LEATHER TECHNOLOGY",
    "MB":"MECHANICAL ENGINEERING (AUTOMOBILE)",
    "ME":"MECHANICAL ENGINEERING",
    "MF":"MECHANICAL ENGINEERING (SS)",
    "MG":"MECHATRONICS (SS)",
    "MJ":"MECHANICAL AND SMART MANUFACTURING",
    "MM":"MECHANICAL ENGINEERING (MANUFACTURING)",
    "MS":"MECHANICAL ENGINEERING (SANDWICH) (SS)",
    "MT":"METALLURGICAL ENGINEERING",
    "MU":"MECHANICAL AND AUTOMATION ENGINEERING",
    "MY":"METALLURGICAL ENGINEERING (SS)",
    "MZ":"MECHATRONICS ENGINEERING",
    "PC":"PETRO CHEMICAL TECHNOLOGY",
    "PD":"PETROLEUM ENGINEERING",
    "PE":"PETROLEUM ENGINEERING AND TECHNOLOGY",
    "PH":"PHARMACEUTICAL TECHNOLOGY",
    "PM":"PHARMACEUTICAL TECHNOLOGY (SS)",
    "PN":"PRODUCTION ENGINEERING (SS)",
    "PP":"PETROLEUM ENGINEERING AND TECHNOLOGY (SS)",
    "PR":"PRODUCTION ENGINEERING",
    "PS":"PRODUCTION ENGINEERING (SANDWICH) (SS)",
    "RA":"ROBOTICS AND AUTOMATION (SS)",
    "RM":"ROBOTICS AND AUTOMATION",
    "RP":"RUBBER AND PLASTIC TECHNOLOGY",
    "SC":"CSE (CYBER SECURITY)",
    "TC":"TEXTILE CHEMISTRY",
    "TT":"TEXTILE TECHNOLOGY (SS)",
    "TX":"TEXTILE TECHNOLOGY",
    "XC":"CIVIL ENGINEERING (TAMIL MEDIUM)",
    "XM":"MECHANICAL ENGINEERING (TAMIL MEDIUM)",
}


def build_lookups(voc_records_all: list[dict]) -> tuple[dict, dict]:
    college_lk = dict(BUILTIN_COLLEGE_NAMES)
    branch_lk  = dict(BUILTIN_BRANCH_NAMES)
    for rec in voc_records_all:
        c = rec["college_code"]
        if rec.get("college_name") and c not in college_lk:
            college_lk[c] = rec["college_name"]
        b = rec["branch_code"]
        if rec.get("branch_name") and b not in branch_lk:
            branch_lk[b] = rec["branch_name"]
    return college_lk, branch_lk


# ─────────────────────────────────────────────────────────────────────────────
#  Cutoff computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cutoffs(rows: list[dict]) -> dict:
    """
    Returns {(college_code, branch_code): {community: min_mark_str}}
    """
    data: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[(r["college_code"], r["branch_code"])][r["community"]].append(r["mark"])

    result = {}
    for key, comm_marks in data.items():
        cutoffs = {}
        for cat, marks in comm_marks.items():
            m = min(marks)
            # Format: drop trailing zeros on decimals
            s = f"{m:.4f}".rstrip("0").rstrip(".")
            cutoffs[cat] = s
        result[key] = cutoffs
    return result


def fmt_mark(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Download
# ─────────────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 5_000:
        print(f"    [cache] {dest.name}")
        return dest
    try:
        print(f"    GET {url[:80]}")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
        if r.status_code == 200 and len(r.content) > 5_000:
            dest.write_bytes(r.content)
            print(f"    ✓ {len(r.content):,} bytes → {dest.name}")
            return dest
        print(f"    ✗ HTTP {r.status_code}")
        return None
    except Exception as e:
        print(f"    ✗ {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(years: list[int], pdf_dir: Path,
        filter_codes: set | None, out_path: Path) -> list[dict]:

    pdf_dir.mkdir(parents=True, exist_ok=True)
    allotment_rows: dict[int, list] = defaultdict(list)
    voc_records:    dict[int, list] = defaultdict(list)

    for (year, label, url) in PDF_SOURCES:
        if year not in years:
            continue
        safe = re.sub(r"[^a-z0-9_]", "_", url.split("/")[-1].split("?")[0].lower())
        dest = pdf_dir / f"{year}_{label}_{safe}"
        print(f"\n  [{year}] {label}")
        pdf_path = download(url, dest)
        if not pdf_path:
            continue

        if label == "voc":
            recs = parse_vocational_pdf(pdf_path)
            print(f"    → {len(recs)} vocational cutoff rows")
            voc_records[year].extend(recs)
        else:
            recs = parse_allotment_pdf(pdf_path)
            print(f"    → {len(recs)} allotment rows parsed")
            allotment_rows[year].extend(recs)

    all_voc_flat = [r for recs in voc_records.values() for r in recs]
    college_lk, branch_lk = build_lookups(all_voc_flat)

    all_records: list[dict] = []

    for year in years:
        print(f"\n  Computing cutoffs for {year}…")

        # ── Academic: MIN mark per (college, branch, community) ──────────
        if allotment_rows[year]:
            rows = allotment_rows[year]
            if filter_codes:
                rows = [r for r in rows if r["college_code"] in filter_codes]
            cutoff_map = compute_cutoffs(rows)
            for (ccode, bcode), cutoffs in sorted(cutoff_map.items(),
                                                   key=lambda x: int(x[0][0])):
                if not cutoffs:
                    continue
                all_records.append({
                    "college_name":     college_lk.get(ccode, f"College {ccode}"),
                    "department":       branch_lk.get(bcode, bcode),
                    "counselling_code": ccode,
                    "cutoffs":          cutoffs,
                    "year":             year,
                    "_src":             "academic",
                })
            print(f"    Academic: {len(cutoff_map):,} college-branch combos")

        # ── Vocational: use pre-computed cutoffs ─────────────────────────
        for rec in voc_records[year]:
            ccode = rec["college_code"]
            if filter_codes and ccode not in filter_codes:
                continue
            bcode = rec["branch_code"]
            all_records.append({
                "college_name":     college_lk.get(ccode, rec.get("college_name", f"College {ccode}")),
                "department":       branch_lk.get(bcode, rec.get("branch_name", bcode)),
                "counselling_code": ccode,
                "cutoffs":          rec["cutoffs"],
                "year":             year,
                "_src":             "vocational",
            })

    # De-duplicate — prefer academic over vocational for same key
    seen:   dict = {}
    unique: list[dict] = []
    for r in all_records:
        key = (r["counselling_code"], r["department"], r["year"])
        if key not in seen:
            seen[key] = len(unique)
            unique.append(r)
        elif r["_src"] == "academic" and unique[seen[key]]["_src"] == "vocational":
            unique[seen[key]] = r

    for r in unique:
        r.pop("_src", None)

    unique.sort(key=lambda r: (int(r["counselling_code"]), r["department"], r["year"]))

    out_path.write_text(json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    print(f"\n{'═'*60}")
    print(f"Total records  : {len(unique):,}")
    print(f"Output         : {out_path.resolve()}")
    for y, cnt in sorted(Counter(r["year"] for r in unique).items()):
        print(f"  {y} : {cnt:,} records")

    ceg = [r for r in unique if r["counselling_code"] == "1"]
    if ceg:
        print(f"\nSample — CEG Campus (code 1), {len(ceg)} entries:")
        for r in sorted(ceg, key=lambda x: (x["department"], x["year"]))[:12]:
            print(f"  [{r['year']}] {r['department'][:45]:46s} {r['cutoffs']}")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="TNEA Cutoff Scraper — computes cutoffs from official allotment PDFs"
    )
    ap.add_argument("--years",   nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--pdf-dir", default="./tnea_pdfs")
    ap.add_argument("--codes",   nargs="+", default=None,
                    help="Filter to specific counselling codes e.g. --codes 1 2 3")
    ap.add_argument("--out",     default="./tnea_cutoffs.json")
    args = ap.parse_args()

    filter_codes = set(args.codes) if args.codes else None

    print("╔══════════════════════════════════════════════════════╗")
    print("║          TNEA Cutoff Mark Scraper  v4                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Years   : {args.years}")
    print(f"  Filter  : {filter_codes or 'all colleges'}")
    print(f"  PDF dir : {Path(args.pdf_dir).resolve()}")
    print(f"  Output  : {Path(args.out).resolve()}")
    print()
    run(years=args.years, pdf_dir=Path(args.pdf_dir),
        filter_codes=filter_codes, out_path=Path(args.out))

if __name__ == "__main__":
    main()