"""
Preprocess 007_gads_paper_draft.md for pandoc → xelatex PDF export.

Fixes applied:
- Extracts title (H1) and abstract (## Abstract section) into YAML frontmatter
- Removes the ## Abstract section from the body (it becomes a LaTeX abstract block)
- Drops manual section numbers from headings so pandoc --number-sections works cleanly
- Marks ## References, ## Appendix as {.unnumbered} so they don't get section numbers
- Writes a clean intermediate .md then calls pandoc
"""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT  = REPO_ROOT / "approach_docs" / "007_gads_paper_draft.md"
TEMP   = REPO_ROOT / "approach_docs" / "_paper_export_tmp.md"
OUTPUT = REPO_ROOT / "approach_docs" / "gads_paper.pdf"


def strip_section_number(heading_text: str) -> str:
    """Remove leading '3.1 ', '6.3 ', 'A.' style numbers from a heading."""
    # Matches: "1. Intro", "3.1 Foo", "6.3 Bar: Baz", "Appendix A: ..."
    return re.sub(r'^(?:\d+\.)*\d+\.?\s+', '', heading_text)


def process(text: str) -> tuple[str, str, str]:
    """
    Returns (title, abstract_text, cleaned_body_markdown).
    """
    lines = text.splitlines()
    title = ""
    abstract_lines: list[str] = []
    body_lines: list[str] = []

    in_abstract = False
    past_title = False

    for line in lines:
        # Extract H1 title
        if not past_title and line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            past_title = True
            continue

        # Detect ## Abstract start
        if re.match(r'^## Abstract\s*$', line, re.IGNORECASE):
            in_abstract = True
            continue

        # Detect next ## section → end of abstract
        if in_abstract and line.startswith("## "):
            in_abstract = False
            # fall through to body processing

        if in_abstract:
            abstract_lines.append(line)
            continue

        # Strip manual numbers from ALL headings
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            hashes = m.group(1)
            heading = m.group(2)
            clean = strip_section_number(heading)

            # Mark unnumbered sections
            unnumbered_patterns = [
                r'^References?\s*$',
                r'^Appendix',
                r'^Acknowledgements?\s*$',
            ]
            is_unnumbered = any(re.match(p, clean, re.IGNORECASE) for p in unnumbered_patterns)
            suffix = " {.unnumbered}" if is_unnumbered else ""
            body_lines.append(f"{hashes} {clean}{suffix}")
        else:
            body_lines.append(line)

    abstract = "\n".join(abstract_lines).strip()
    body = "\n".join(body_lines)
    return title, abstract, body


def build_yaml_frontmatter(title: str, abstract: str) -> str:
    # Escape any double-quotes in title/abstract for YAML block scalar
    abstract_indented = textwrap.indent(abstract, "  ")
    return f"""---
title: |
  {title}
abstract: |
{abstract_indented}
geometry: "margin=1in"
fontsize: 11pt
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\renewcommand{{\\abstractname}}{{Abstract}}
---
"""


def main():
    text = INPUT.read_text(encoding="utf-8")
    title, abstract, body = process(text)

    if not title:
        print("ERROR: could not find H1 title in source", file=sys.stderr)
        sys.exit(1)

    frontmatter = build_yaml_frontmatter(title, abstract)
    TEMP.write_text(frontmatter + "\n" + body, encoding="utf-8")
    print(f"Preprocessed → {TEMP}")

    cmd = [
        "pandoc", str(TEMP),
        "--pdf-engine=xelatex",
        "--shift-heading-level-by=-1",     # ## → \section, ### → \subsection (title is in YAML, body starts at ##)
        f"--output={OUTPUT}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("pandoc stderr:\n", result.stderr)
        sys.exit(result.returncode)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"✅  Written: {OUTPUT}  ({size_kb:.0f} KB)")

    # Keep temp file for inspection (delete manually)


if __name__ == "__main__":
    main()
