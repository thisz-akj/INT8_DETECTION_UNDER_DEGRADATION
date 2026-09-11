"""One-time surgical patch: insert the FP32-vs-INT8 AP-by-size table into
results/summary_report.docx, right after the per-class AP table (Table 2),
renumbering the tables that follow. Leaves everything else untouched."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PATH = "results/summary_report.docx"

INK = RGBColor(0x14, 0x18, 0x1B)
DIM = RGBColor(0x4A, 0x52, 0x59)
HEADER_BG = "2E6F95"

ROWS = [
    ("Small (<32²px)", "0.2045", "0.1658", "-0.0387 (-18.9%)"),
    ("Medium (32²-96²px)", "0.5978", "0.5730", "-0.0248 (-4.1%)"),
    ("Large (>96²px)", "0.7186", "0.7518", "+0.0331 (+4.6%)"),
]
HEADERS = ["Size", "FP32 AP@[.5:.95]", "INT8 AP@[.5:.95]", "Delta"]
COL_WIDTHS_CM = [4.2, 3.5, 3.5, 4.0]

CAPTION = (
    "Small-object AP takes the largest relative hit (-18.9%) while large objects are "
    "essentially unaffected (+4.6%, within noise) -- 8-bit resolution has the least "
    "room to spare exactly where the signal is already thinnest."
)


def set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def style_cell(cell, text, bold=False, color=None, size=10, align_center=False):
    cell.text = ""
    p = cell.paragraphs[0]
    if align_center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    if color:
        run.font.color.rgb = color


def rename_in_runs(paragraph, old, new):
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new, 1)
            return True
    return False


doc = Document(PATH)

# Rename the two existing headings that come after the per-class table, so
# there's no ambiguity once we insert the new "Table 3." in between.
renamed = 0
for p in doc.paragraphs:
    t = p.text.strip()
    if t.startswith("Table 3. mAP@0.5 across all five conditions"):
        if not rename_in_runs(p, "Table 3.", "Table 4."):
            raise SystemExit("Could not rename 'Table 3.' -- aborting, nothing saved.")
        renamed += 1
    elif t.startswith("Table 4. Task 3 intervention"):
        if not rename_in_runs(p, "Table 4.", "Table 5."):
            raise SystemExit("Could not rename 'Table 4.' -- aborting, nothing saved.")
        renamed += 1
if renamed != 2:
    raise SystemExit(f"Expected to rename 2 headings, renamed {renamed} -- aborting, nothing saved.")

# Anchor: the (renamed) "Table 4. mAP@0.5..." heading -- insert the new table before it.
anchor = None
for p in doc.paragraphs:
    if p.text.strip().startswith("Table 4. mAP@0.5"):
        anchor = p
        break
if anchor is None:
    raise SystemExit("Could not find the renamed 'Table 4.' heading -- aborting, nothing saved.")

heading = anchor.insert_paragraph_before("Table 3. AP by object size, FP32 vs INT8 (clean images).")
heading.paragraph_format.space_before = Pt(4)
heading.paragraph_format.space_after = Pt(6)
for run in heading.runs:
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = INK

caption_p = anchor.insert_paragraph_before("")
caption_p.paragraph_format.space_after = Pt(10)
crun = caption_p.add_run(CAPTION)
crun.italic = True
crun.font.size = Pt(9)
crun.font.color.rgb = DIM

table = doc.add_table(rows=1 + len(ROWS), cols=len(HEADERS))
table.alignment = WD_TABLE_ALIGNMENT.LEFT
table.autofit = False

for j, h in enumerate(HEADERS):
    cell = table.rows[0].cells[j]
    style_cell(cell, h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=10, align_center=(j > 0))
    set_cell_shading(cell, HEADER_BG)

for i, row in enumerate(ROWS):
    for j, val in enumerate(row):
        cell = table.rows[i + 1].cells[j]
        style_cell(cell, val, bold=False, color=INK, size=10, align_center=(j > 0))

for row in table.rows:
    for j, w in enumerate(COL_WIDTHS_CM):
        row.cells[j].width = Cm(w)

heading._p.addnext(table._tbl)
table._tbl.addnext(caption_p._p)

doc.save(PATH)
print("Patched", PATH)
