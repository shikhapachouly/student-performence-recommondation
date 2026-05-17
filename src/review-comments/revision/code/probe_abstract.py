"""Look for the Abstract paragraph in the revised .docx."""
import os
from docx import Document

DOCX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "Revised_Paper_20262542.docx",
)

doc = Document(DOCX)
for i, p in enumerate(doc.paragraphs[:25]):
    text = (p.text or "")[:120]
    red = sum(1 for r in p.runs if r.font.color and r.font.color.rgb and str(r.font.color.rgb).upper() == "C00000")
    print(f"para {i:3d} [{red} red runs]: {text!r}")
