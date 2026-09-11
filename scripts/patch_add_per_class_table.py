"""One-time surgical patch: insert a per-class AP table into the user-edited
results/summary_report.docx, right after Table 1, without touching anything else."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PATH = "results/summary_report.docx"

INK = RGBColor(0x14, 0x18, 0x1B)
HEADER_BG = "2E6F95"

ROWS = [
    ("person", "0.7751", "0.7630", "0.5291", "0.5111"),
    ("bicycle", "0.4081", "0.3885", "0.2446", "0.2356"),
    ("car", "0.5937", "0.5879", "0.3797", "0.3704"),
    ("traffic light", "0.4361", "0.4284", "0.2326", "0.2138"),
    ("stop sign", "0.7178", "0.6761", "0.6872", "0.6411"),
]
HEADERS = ["Class", "FP32 AP@0.5", "INT8 AP@0.5", "FP32 AP@0.5:0.95", "INT8 AP@0.5:0.95"]
COL_WIDTHS_CM = [3.2, 2.7, 2.7, 3.1, 3.1]


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


doc = Document(PATH)

# Rename the two existing table headings FIRST (by their full, unambiguous current
# text) so there's no confusion with the new "Table 2." we're about to insert.
def rename_in_runs(paragraph, old, new):
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new, 1)
            return True
    return False

renamed = 0
for p in doc.paragraphs:
    t = p.text.strip()
    if t.startswith("Table 2. mAP@0.5 across all five conditions"):
        if not rename_in_runs(p, "Table 2.", "Table 3."):
            raise SystemExit("Could not find 'Table 2.' text within a single run -- aborting, nothing saved.")
        renamed += 1
    elif t.startswith("Table 3. Task 3 intervention"):
        if not rename_in_runs(p, "Table 3.", "Table 4."):
            raise SystemExit("Could not find 'Table 3.' text within a single run -- aborting, nothing saved.")
        renamed += 1
if renamed != 2:
    raise SystemExit(f"Expected to rename 2 headings, renamed {renamed} -- aborting, nothing saved.")

# Now find the (renamed) "Table 3. mAP@0.5..." paragraph -- the anchor to insert before.
anchor = None
for p in doc.paragraphs:
    if p.text.strip().startswith("Table 3. mAP@0.5"):
        anchor = p
        break
if anchor is None:
    raise SystemExit("Could not find the renamed 'Table 3.' heading paragraph -- aborting, nothing saved.")

# 1) new heading paragraph, inserted right before the anchor
heading = anchor.insert_paragraph_before("Table 2. Per-class AP, FP32 vs INT8 (clean images).")
heading.paragraph_format.space_before = Pt(4)
heading.paragraph_format.space_after = Pt(6)
for run in heading.runs:
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = INK

# 2) blank spacer paragraph after the table (also inserted before anchor, so it ends up
#    table -> spacer -> anchor once the table itself is moved into place below)
spacer = anchor.insert_paragraph_before("")

# 3) build the table by appending to the doc, then relocate its XML element into place
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

# move the table's XML element from the end of the doc to right after `heading`
heading._p.addnext(table._tbl)
# ensure the spacer paragraph sits right after the (now relocated) table
table._tbl.addnext(spacer._p)

doc.save(PATH)
print("Patched", PATH)
