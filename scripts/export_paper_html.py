"""Convert approach_docs/007_gads_paper_draft.md to a print-ready HTML file."""
import re
import sys
from pathlib import Path
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / "approach_docs" / "007_gads_paper_draft.md"
OUTPUT = REPO_ROOT / "approach_docs" / "gads_paper.html"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Linux+Libertine+O&family=Linux+Biolinum+O&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: "Linux Libertine O", "Georgia", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #111;
    background: #fff;
    max-width: 720px;
    margin: 0 auto;
    padding: 40px 48px;
}

h1 { font-size: 20pt; margin: 0 0 6px 0; line-height: 1.25; }
h2 { font-size: 14pt; margin: 28px 0 6px 0; border-bottom: 1px solid #bbb; padding-bottom: 3px; }
h3 { font-size: 12pt; margin: 20px 0 4px 0; font-style: italic; font-weight: bold; }
h4 { font-size: 11pt; margin: 16px 0 4px 0; font-weight: bold; }

p { margin: 8px 0; }
ul, ol { margin: 8px 0 8px 24px; }
li { margin: 3px 0; }

strong { font-weight: bold; }
em { font-style: italic; }

/* Abstract block */
.abstract {
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 12px 16px;
    margin: 16px 0 24px 0;
    font-size: 10.5pt;
    background: #fafafa;
}
.abstract-label {
    font-weight: bold;
    font-variant: small-caps;
    display: block;
    margin-bottom: 6px;
}

code {
    font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
}

pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 10px 14px;
    overflow-x: auto;
    margin: 10px 0;
    font-size: 8.5pt;
    line-height: 1.45;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9.5pt;
}
th {
    background: #e8e8e8;
    border: 1px solid #bbb;
    padding: 5px 8px;
    text-align: left;
    font-weight: bold;
}
td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f9f9f9; }

blockquote {
    border-left: 3px solid #bbb;
    padding: 4px 12px;
    margin: 8px 0;
    color: #555;
    font-style: italic;
}

a { color: #1a4a8a; text-decoration: none; }
a:hover { text-decoration: underline; }

hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }

/* Section headings with numbers look better in small-caps */
h2 { font-variant: normal; }

@media print {
    body { padding: 0; max-width: 100%; font-size: 10.5pt; }
    h2 { page-break-before: auto; }
    pre, table { page-break-inside: avoid; }
    a { color: inherit; }
    .abstract { border: 1px solid #999; }
}
"""

def build_html(md_text: str) -> str:
    # Separate title / abstract from body
    lines = md_text.split("\n")

    # Strip leading "# " from title line
    title = ""
    body_lines = []
    in_abstract = False
    abstract_lines = []
    past_abstract = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if not title and line.startswith("# "):
            title = line[2:].strip()
            i += 1
            continue
        # Treat the content between the title and "## 1. Introduction" as the abstract
        if title and not past_abstract and not line.startswith("## "):
            abstract_lines.append(line)
            i += 1
            continue
        if not past_abstract and line.startswith("## "):
            past_abstract = True
        body_lines.append(line)
        i += 1

    abstract_md = "\n".join(abstract_lines).strip()
    body_md = "\n".join(body_lines)

    extensions = [TableExtension(), FencedCodeExtension(), "md_in_html"]
    md = markdown.Markdown(extensions=extensions)

    abstract_html = md.convert(abstract_md)
    md.reset()
    body_html = md.convert(body_md)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{title}</h1>
<div class="abstract">
  <span class="abstract-label">Abstract</span>
  {abstract_html}
</div>
{body_html}
</body>
</html>
"""


def main():
    md_text = INPUT.read_text(encoding="utf-8")
    html = build_html(md_text)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Written: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
