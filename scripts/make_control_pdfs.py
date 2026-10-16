"""Generate three control PDFs with a *known* structure, and ``expected.json`` describing it.

The PDFs carry PDF bookmarks and numbered headings (what Docling's heading hierarchy uses),
real tables, real figures (matplotlib charts) and one "needle" sentence each that a query must
retrieve. ``hgrag verify`` compares the pipeline output with ``expected.json``.

    uv run python scripts/make_control_pdfs.py            # writes data/pdf_control/
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "pdf_control"
ss = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Times-Roman", fontSize=11, leading=15, spaceAfter=8)
CAPTION = ParagraphStyle("Cap", parent=BODY, fontName="Times-Italic", fontSize=9.5, alignment=1)
HEAD = {1: ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24, spaceBefore=16, spaceAfter=10),
        2: ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceBefore=12, spaceAfter=6),
        3: ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=4)}
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=26, leading=30, spaceAfter=18)


class Doc(BaseDocTemplate):
    """Adds a PDF bookmark (outline entry) for every heading: Docling reads them for levels."""

    def __init__(self, path, title, **kw):
        super().__init__(str(path), pagesize=A4, title=title, author="hgrag control", invariant=1,
                         leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._footer)])
        self._n = 0

    @staticmethod
    def _footer(canv, doc):
        canv.setFont("Helvetica", 8)
        canv.drawCentredString(A4[0] / 2, 1.2 * cm, f"page {doc.page}")

    def afterFlowable(self, fl):
        lvl = getattr(fl, "_hlevel", None)
        if lvl:
            key = f"h{self._n}"
            self._n += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(fl.getPlainText(), key, level=lvl - 1, closed=False)


def H(level: int, text: str) -> Paragraph:
    p = Paragraph(text, HEAD[level])
    p._hlevel = level
    return p


def P(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def chart(kind: str, title: str, xs, ys, ylabel: str) -> Image:
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=140)
    (ax.bar if kind == "bar" else ax.plot)(xs, ys, **({"color": "#5B93E6"} if kind == "bar" else {"marker": "o", "color": "#2F8F5B"}))
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=12 * cm, height=12 * cm * 2.8 / 5.2)


def table(rows, widths=None) -> Table:
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEBFF")),
                           ("GRID", (0, 0), (-1, -1), .6, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 10), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    return t


def fig_block(img: Image, caption: str):
    return KeepTogether([img, Paragraph(caption, CAPTION), Spacer(1, 8)])


# ----------------------------------------------------------------------------- document 1
def aurora(path: Path) -> dict:
    s = [Paragraph("Aurora Bank Annual Report 2025", TITLE)]
    s += [H(1, "1 Executive summary"),
          P("Aurora Bank closed 2025 with a net profit of 412 million euros, a rise of nine percent over the previous "
            "year. Retail deposits grew steadily while the corporate lending book stayed flat. The board proposes a "
            "dividend of 1.80 euros per share and confirms its commitment to a cautious risk policy."),
          H(1, "2 Financial results"),
          H(2, "2.1 Revenue"),
          P("Total revenue reached 1,380 million euros. Net interest income remained the main contributor, "
            "followed by fees from asset management and by payment services, which benefited from the growth "
            "of contactless transactions across all regions."),
          table([["Revenue line", "2024 (M EUR)", "2025 (M EUR)"], ["Net interest income", "720", "781"],
                 ["Fees and commissions", "398", "431"], ["Other income", "151", "168"]], [6 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 10),
          H(2, "2.2 Costs"),
          P("Operating costs increased by only two percent thanks to the consolidation of back-office functions. "
            "The cost to income ratio fell to fifty-eight percent, its lowest level in ten years, while investment "
            "in cybersecurity doubled compared with 2024."),
          H(1, "3 Branch network"),
          H(2, "3.1 New branches"),
          P("The bank opened three new branches in the Alsace region during 2025, in Colmar, Mulhouse and Haguenau. "
            "Each branch combines a traditional counter with a digital advice area and is staffed by advisers "
            "trained in sustainable finance."),
          fig_block(chart("bar", "Branches per year", ["2021", "2022", "2023", "2024", "2025"], [212, 208, 205, 201, 204],
                          "branches"), "Figure 1. Number of Aurora Bank branches per year."),
          H(2, "3.2 Closures"),
          P("Two small rural branches were merged into neighbouring ones. Customers were informed six months in advance "
            "and every affected employee was offered a position within a radius of thirty kilometres.")]
    Doc(path, "Aurora Bank Annual Report 2025").build(s)
    return {"headings": [["1 Executive summary", 1], ["2 Financial results", 1], ["2.1 Revenue", 2], ["2.2 Costs", 2],
                         ["3 Branch network", 1], ["3.1 New branches", 2], ["3.2 Closures", 2]],
            "n_figures": 1, "n_tables": 1,
            "needles": [{"query": "three new branches in the Alsace region", "section": "3.1 New branches"},
                        {"query": "cost to income ratio lowest level in ten years", "section": "2.2 Costs"}]}


# ----------------------------------------------------------------------------- document 2
def botany(path: Path) -> dict:
    s = [Paragraph("Field Guide to Alpine Botany", TITLE)]
    s += [H(1, "1 Introduction"),
          P("Alpine plants live in a harsh world of wind, cold and intense ultraviolet light. This guide describes "
            "the species most often met by hikers between the forest limit and the permanent snow, and explains "
            "how each of them survives the short mountain summer."),
          H(1, "2 Plant communities"),
          H(2, "2.1 Limestone slopes"),
          H(3, "2.1.1 Edelweiss"),
          P("Edelweiss grows on limestone above 1800 metres, in sunny and well drained places. Its woolly white "
            "bracts protect the tiny flowers from drying winds and from radiation, and the plant is protected by law "
            "in most alpine countries."),
          H(3, "2.1.2 Alpine gentian"),
          P("The alpine gentian forms low cushions covered with deep blue flowers in early summer. It prefers "
            "short grass on calcareous soil and flowers a few days after the snow has melted."),
          fig_block(chart("line", "Flowering altitude by month", ["Jun", "Jul", "Aug", "Sep"], [1800, 2100, 2400, 2200],
                          "altitude (m)"), "Figure 1. Mean altitude of flowering gentians through the summer."),
          H(2, "2.2 Siliceous screes"),
          P("On acidic rock the vegetation is dominated by cushion plants such as moss campion. Their dense "
            "shape traps heat and moisture and lets them colonise ground that moves slowly with frost and thaw."),
          H(1, "3 Observation tips"),
          H(2, "3.1 Best season"),
          P("The best period for observation runs from late June to the middle of August, depending on altitude "
            "and on the date of snow melt. Early morning light is ideal for photography and for counting flowers."),
          fig_block(chart("bar", "Species seen per hiking day", ["Jun", "Jul", "Aug", "Sep"], [9, 21, 26, 12],
                          "species"), "Figure 2. Average number of species recorded per hiking day."),
          H(2, "3.2 Respecting the habitat"),
          P("Stay on marked paths, never pick protected species and keep dogs on a lead. Trampled cushions need "
            "decades to recover, so a few careless steps can undo the growth of a whole generation of plants.")]
    Doc(path, "Field Guide to Alpine Botany").build(s)
    return {"headings": [["1 Introduction", 1], ["2 Plant communities", 1], ["2.1 Limestone slopes", 2],
                         ["2.1.1 Edelweiss", 3], ["2.1.2 Alpine gentian", 3], ["2.2 Siliceous screes", 2],
                         ["3 Observation tips", 1], ["3.1 Best season", 2], ["3.2 Respecting the habitat", 2]],
            "n_figures": 2, "n_tables": 0,
            "needles": [{"query": "edelweiss grows on limestone above 1800 metres", "section": "2.1.1 Edelweiss"},
                        {"query": "cushion plants colonise ground that moves with frost", "section": "2.2 Siliceous screes"}]}


# ----------------------------------------------------------------------------- document 3
def netops(path: Path) -> dict:
    s = [Paragraph("Network Operations Handbook", TITLE)]
    s += [H(1, "1 Scope"),
          P("This handbook defines how the operations team monitors, maintains and restores the production network. "
            "It applies to every datacentre, to the wide area links and to the services that depend on them.")]
    s += [H(1, "2 Monitoring"),
          H(2, "2.1 Alert levels"),
          P("Alerts are classified in three levels. A warning requires attention during office hours, a critical "
            "alert must be acknowledged within fifteen minutes, and a major incident triggers the on-call manager."),
          table([["Level", "Acknowledge within", "Escalation"], ["Warning", "4 hours", "Team lead"],
                 ["Critical", "15 minutes", "On-call engineer"], ["Major", "5 minutes", "On-call manager"]],
                [4 * cm, 5 * cm, 5 * cm]),
          Spacer(1, 10),
          H(2, "2.2 Health checks"),
          P("Every service exposes a health endpoint which the monitoring system queries each thirty seconds. "
            "The checks cover the following points:"),
          ListFlowable([ListItem(Paragraph(t, BODY)) for t in
                        ["link status and packet loss on every uplink", "latency between the datacentres",
                         "certificate expiry dates", "disk usage of the log collectors"]], bulletType="bullet"),
          H(1, "3 Recovery"),
          H(2, "3.1 Failover"),
          P("If the primary datacentre becomes unreachable, traffic must failover to the secondary datacentre within "
            "ninety seconds. The procedure is automatic, and the operator only verifies the result with the command below."),
          Preformatted("$ netctl failover status --site secondary\nactive: true\nlag_seconds: 12", ss["Code"]),
          H(2, "3.2 Post-incident review"),
          P("A written review is produced within five working days after each major incident. It lists the timeline, the "
            "root cause, the customer impact and the corrective actions, each with an owner and a due date.")]
    Doc(path, "Network Operations Handbook").build(s)
    return {"headings": [["1 Scope", 1], ["2 Monitoring", 1], ["2.1 Alert levels", 2], ["2.2 Health checks", 2],
                         ["3 Recovery", 1], ["3.1 Failover", 2], ["3.2 Post-incident review", 2]],
            "n_figures": 0, "n_tables": 1,
            "needles": [{"query": "failover to the secondary datacentre within ninety seconds", "section": "3.1 Failover"},
                        {"query": "critical alert acknowledged within fifteen minutes", "section": "2.1 Alert levels"}]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, fn in (("01_aurora_bank_annual_report.pdf", aurora), ("02_alpine_botany_field_guide.pdf", botany),
                     ("03_network_operations_handbook.pdf", netops)):
        expected[name] = fn(OUT / name)
        print("wrote", OUT / name)
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
