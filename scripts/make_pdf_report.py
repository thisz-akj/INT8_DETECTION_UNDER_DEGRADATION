"""Generate a 2-page PDF summary: what we did, results tables, surprises, next steps."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 ListFlowable, ListItem)

INK = colors.HexColor("#14181B")
DIM = colors.HexColor("#4A5259")
ACCENT = colors.HexColor("#2E6F95")
INT8C = colors.HexColor("#C97A24")
LINE = colors.HexColor("#D8DCD9")
HILITE = colors.HexColor("#FBF1E4")

OUT = "results/two_page_summary.pdf"

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=20,
                             textColor=INK, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9, leading=12,
                                textColor=DIM, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=13,
                          textColor=INK, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.7, leading=11.8,
                            textColor=INK, spaceAfter=3),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.1, leading=10.8,
                             textColor=INK),
    "tablecap": ParagraphStyle("tablecap", fontName="Helvetica-Oblique", fontSize=7.6, leading=9.2,
                                textColor=DIM, spaceBefore=1.5, spaceAfter=5),
}

def table(data, col_widths, highlight_rows=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]
    for r in (highlight_rows or []):
        style.append(("BACKGROUND", (0, r), (-1, r), HILITE))
    t.setStyle(TableStyle(style))
    return t


def bullets(items, style="small"):
    return ListFlowable(
        [ListItem(Paragraph(i, styles[style]), spaceAfter=2) for i in items],
        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=7,
    )


doc = SimpleDocTemplate(OUT, pagesize=A4,
                         topMargin=13 * mm, bottomMargin=12 * mm,
                         leftMargin=18 * mm, rightMargin=18 * mm)

flow = []
flow.append(Paragraph("INT8 Detection Under Degradation", styles["title"]))
flow.append(Paragraph(
    "YOLOv8n, COCO val2017 (5 classes, 500-image eval set) &nbsp;|&nbsp; "
    "12th Gen Intel Core i5-1245U, single-threaded", styles["subtitle"]))

# ---------- What we did ----------
flow.append(Paragraph("What we did", styles["h2"]))
flow.append(Paragraph(
    "Picked YOLOv8n for its mature ONNX/INT8 tooling, then built a fixed 500-image COCO val2017 "
    "subset restricted to person/car/bicycle/traffic light/stop sign (2,353 ground-truth boxes). "
    "<b>Task 1</b>: exported to ONNX, measured the FP32 baseline, then applied post-training static "
    "INT8 quantization via ONNX Runtime (QDQ, U8S8, per-channel, Conv-only &mdash; the detection head's "
    "final box/score concat has to stay in float or every class score gets quantized to zero), calibrated "
    "on a disjoint 120-image set. <b>Task 2</b>: built four degraded copies of the eval set (motion blur, "
    "low light, JPEG q30, downscale/upscale) and ran both precisions across all of them. <b>Task 3</b>: "
    "picked the worst degradation (motion blur) and tried one targeted fix &mdash; recalibrating INT8 on "
    "blurred images instead of clean ones &mdash; to see if it closed the gap.",
    styles["body"]))

# ---------- Results tables ----------
flow.append(Paragraph("Results", styles["h2"]))

flow.append(table(
    [["FP32 vs INT8", "FP32", "INT8", "Δ"],
     ["mAP@0.5", "0.586", "0.569", "-3.0%"],
     ["mAP@0.5:0.95", "0.415", "0.394", "-4.9%"],
     ["Latency, mean (1 thread)", "134.9 ms", "69.7 ms", "1.94x faster"],
     ["Model size", "12.85 MB", "3.60 MB", "3.57x smaller"]],
    col_widths=[55 * mm, 30 * mm, 30 * mm, 35 * mm],
))
flow.append(Paragraph("Table 1 &mdash; accuracy, latency, and size, clean images.", styles["tablecap"]))

flow.append(table(
    [["mAP@0.5 by condition", "FP32", "INT8", "INT8 rel. gap"],
     ["Low light (γ=2.5)", "0.516", "0.512", "-0.6%"],
     ["Clean", "0.586", "0.569", "-3.0%"],
     ["JPEG q30", "0.534", "0.514", "-3.6%"],
     ["Downscale 50% → upscale", "0.543", "0.517", "-4.8%"],
     ["Motion blur (15x15 kernel)", "0.281", "0.250", "-10.9%"]],
    col_widths=[55 * mm, 30 * mm, 30 * mm, 35 * mm],
    highlight_rows=[5],
))
flow.append(Paragraph(
    "Table 2 &mdash; the 5x2 degradation table. “Rel. gap” = INT8's shortfall as a fraction of "
    "FP32's own score in that condition, so it's comparable across rows.", styles["tablecap"]))

flow.append(table(
    [["Blur-recalibration intervention", "On motion blur", "On clean images"],
     ["FP32", "0.281", "0.586"],
     ["INT8, clean-calibrated (original)", "0.250", "0.569"],
     ["INT8, blur-calibrated (new)", "0.250", "0.582"]],
    col_widths=[68 * mm, 41 * mm, 41 * mm],
    highlight_rows=[3],
))
flow.append(Paragraph(
    "Table 3 &mdash; mAP@0.5. Recalibrating on blurred images left the target condition unchanged "
    "but closed 78% of the clean-image gap instead. Cost: +6 bytes, &lt;0.3% latency difference.",
    styles["tablecap"]))

# ---------- What surprised us ----------
flow.append(Paragraph("What surprised us", styles["h2"]))
flow.append(bullets([
    "The very first INT8 build was <b>slower</b> than FP32 (177ms vs 135ms) &mdash; S8S8 quantization "
    "doesn't hit ONNX Runtime's accelerated x86-64 kernels, only U8S8 does. Same hardware, one dtype flag.",
    "Quantizing the model's own output concat (box coords ~0&ndash;640 merged with 0&ndash;1 class scores "
    "under one shared scale) silently zeroed every detection. Nothing crashed &mdash; it just stopped finding anything.",
    "INT8's accuracy penalty relative to FP32 doesn't scale uniformly with how <i>bad</i> a degradation looks: "
    "it nearly quadruples under motion blur (-3.0% &rarr; -10.9%) but is flat-to-better under low light, "
    "which is arguably the more visually dramatic degradation.",
    "Recalibrating specifically for the failure case (blur) didn't fix that case at all &mdash; it fixed the "
    "case we weren't trying to fix, by tightening quantization scales in a way that helped the typical "
    "(clean) activation distribution more than the one it was calibrated on.",
], style="small"))

# ---------- Next week ----------
flow.append(Paragraph("What we'd do with another week", styles["h2"]))
flow.append(bullets([
    "Try mixed precision instead of a calibration swap: keep the earliest backbone layers "
    "(model.1/model.2 &mdash; the ones with the largest scale shrinkage under blur-calibration) in FP32, "
    "since that's where blur actually destroys the edge information those layers extract.",
    "Quantization-aware training with motion blur in the augmentation pipeline, so the weights adapt "
    "to the degradation instead of only the quantization ranges being recalibrated after the fact.",
    "Widen the eval set to the full 5-class COCO val2017 pool (~3,000 images) &mdash; some per-class/size "
    "splits are thin right now (stop sign has 15 instances total), so those numbers carry real noise.",
    "Test more degradations that plausibly compound with blur in deployment (sensor noise, low-res "
    "capture, weather), and benchmark OpenVINO's INT8 path on the same CPU as a second quantization backend.",
], style="small"))

doc.build(flow)
print("Wrote", OUT)
