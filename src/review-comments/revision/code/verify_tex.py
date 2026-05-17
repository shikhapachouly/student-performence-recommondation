"""Static syntax sanity-check for Response_Letter_20262542.tex.

Catches the most common LaTeX hand-rolling errors before Overleaf does:
    - unbalanced braces
    - unbalanced \\begin{env}/\\end{env}
    - obvious smart-quote / em-dash issues
    - missing \\documentclass or \\begin{document}/\\end{document}

Not a real LaTeX parser — it only flags the bugs we've actually seen.
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "Response_Letter_20262542.tex")

with open(TEX, encoding="utf-8") as f:
    src = f.read()

errors = []
warnings = []

# ---- 1. Required headers
if not re.search(r"^\\documentclass", src, re.MULTILINE):
    errors.append("Missing \\documentclass")
if "\\begin{document}" not in src:
    errors.append("Missing \\begin{document}")
if "\\end{document}" not in src:
    errors.append("Missing \\end{document}")

# ---- 2. Balanced \\begin / \\end environments
begins = re.findall(r"\\begin\{([^}]+)\}", src)
ends = re.findall(r"\\end\{([^}]+)\}", src)
from collections import Counter

bcount = Counter(begins)
ecount = Counter(ends)
for env in set(bcount) | set(ecount):
    if bcount[env] != ecount[env]:
        errors.append(f"Unbalanced environment '{env}': {bcount[env]} begins vs {ecount[env]} ends")

# ---- 3. Balanced {} (rough heuristic — not strict, but flags glaring imbalances)
# Strip TeX comments and verbatim blocks
nocomments = re.sub(r"(?<!\\)%.*", "", src)
verb_re = re.compile(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", re.DOTALL)
noverb = verb_re.sub("", nocomments)
opens = noverb.count("{")
closes = noverb.count("}")
if abs(opens - closes) > 2:  # tolerate 1-2 false positives from \\{ in math
    errors.append(f"Brace imbalance: {opens} '{{' vs {closes} '}}'")

# ---- 4. Forbidden ASCII smart-quote shapes (would break compilation under T1)
suspect_chars = {
    "‘": "left single quote (U+2018)",
    "’": "right single quote (U+2019)",
    "“": "left double quote (U+201C)",
    "”": "right double quote (U+201D)",
}
for ch, label in suspect_chars.items():
    if ch in src:
        # These are fine when inputenc=utf8 is loaded, just note them.
        warnings.append(f"Found {label}: '{ch}' — should render fine with inputenc=utf8")

# ---- 5. Common typo: \\sectref usage
sectref_uses = re.findall(r"\\sectref\{([^}]*)\}", src)
empty_sectref = [s for s in sectref_uses if not s.strip()]
if empty_sectref:
    errors.append(f"Empty \\sectref{{}} arguments: {len(empty_sectref)}")

# ---- 6. Stray BOM
if src.startswith("﻿"):
    warnings.append("File begins with BOM — strip it before submitting")

# ---- 7. Tab characters inside source
if "\t" in src:
    warnings.append("Tab characters present — consider expanding to spaces")

# ---- 8. Length / size sanity
chars = len(src)
lines = src.count("\n") + 1
print(f"File: {TEX}")
print(f"Size: {chars:,} chars  /  {lines:,} lines")
print(f"Begin envs: {len(begins)}, End envs: {len(ends)}")

print()
if not errors and not warnings:
    print("OK — no issues detected.")
elif not errors:
    print("OK — no errors. Warnings (informational):")
    for w in warnings:
        print(f"  - {w}")
else:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    sys.exit(1)
