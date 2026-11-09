"""Generate the synthetic part of the evaluation set: the 3 control PDFs and 5 PDFs built on the known weak points.

Every document is fictitious and generated here, so the set can ship with the kit (no licence issue, unlike Open RAG
Bench). Each new PDF exercises what went wrong on real papers:

* 04: values readable only on a chart, a table spread over pages, a footnote;
* 05: a formula, and a long paragraph set in a heading style (the 180-word false heading of a real paper);
* 06: two columns and unnumbered headings;
* 07: roman numbering with lettered sub-sections, a bibliography (not indexed) and an appendix;
* 08: numbered procedure steps (read as headings by Docling), a code block, a question across two sections.

``expected.json`` describes the real headings of each PDF; the questions are in ``eval_dataset.jsonl`` (written by hand,
not by this script). The two arXiv papers of the set are listed in ``SOURCES.md``.

    uv run python scripts/make_eval_pdfs.py            # writes data/pdf_eval/
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_control_pdfs as ctl  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend chosen by make_control_pdfs)
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Frame, Image, ListFlowable, ListItem, PageTemplate, Paragraph, Preformatted,  # noqa: E402
                                Spacer, Table, TableStyle)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "pdf_eval"
H, P, BODY, CAPTION, TITLE = ctl.H, ctl.P, ctl.BODY, ctl.CAPTION, ctl.TITLE
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=8.5, leading=11)
CODE = ParagraphStyle("CodeBlock", fontName="Courier", fontSize=9, leading=12, leftIndent=12)


def bar_chart(title: str, labels, values, ylabel: str) -> Image:
    """A bar chart whose values are printed on the bars only: the text of the page does not give them."""
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=140)
    bars = ax.bar(labels, values, color="#5B93E6")
    ax.bar_label(bars, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    return _image(fig, 12, 2.8 / 5.2)


def formula(tex: str) -> Image:
    fig = plt.figure(figsize=(4.0, 0.7), dpi=200)
    fig.text(0.5, 0.5, tex, fontsize=16, ha="center", va="center")
    return _image(fig, 8, 0.7 / 4.0)


def _image(fig, width_cm: float, ratio: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=width_cm * cm * ratio)


def long_table(rows, widths) -> Table:
    """Header row repeated on each page the table spreads over."""
    t = Table(rows, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEBFF")),
                           ("GRID", (0, 0), (-1, -1), .6, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    return t


class TwoColumns(ctl.Doc):
    def __init__(self, path, title):
        super().__init__(path, title)
        gap, w = 0.8 * cm, (self.width - 0.8 * cm) / 2
        frames = [Frame(self.leftMargin, self.bottomMargin, w, self.height, id="left"),
                  Frame(self.leftMargin + w + gap, self.bottomMargin, w, self.height, id="right")]
        self.pageTemplates = []
        self.addPageTemplates([PageTemplate(id="two", frames=frames, onPage=self._footer)])


# ----------------------------------------------------------------------------- document 4
def lumen(path: Path) -> dict:
    batches = [["Batch", "Site", "Units", "Defect rate (%)"]]
    sites = ("Brest", "Lyon", "Turin")
    for i in range(48):
        number = 101 + i
        rate = "2.9" if number == 137 else f"{0.6 + (i * 7 % 13) / 10:.1f}"
        batches.append([f"B-{number}", sites[i % 3], f"{1800 + (i * 37 % 400)}", rate])
    s = [Paragraph("Lumen Labs Quarterly Operations Review, Q3 2026", TITLE)]
    s += [H(1, "1 Overview"),
          P("Lumen Labs manufactures optical sensors for agricultural drones in three plants. The third quarter was "
            "marked by the launch of the LX-4 sensor and by a heat wave that slowed production in August."),
          H(1, "2 Production"),
          H(2, "2.1 Output by site"),
          P("Lyon remained the largest site of the group, while Turin was still ramping up its second assembly line. "
            "The output of each plant is shown in the chart below."),
          ctl.fig_block(bar_chart("Units produced in Q3 2026", list(sites), [340, 515, 280], "thousand units"),
                        "Figure 1. Units produced per site in the third quarter, in thousands."),
          H(2, "2.2 Defect rates by batch"),
          P("Every batch is inspected before shipping. A defect rate above 2.5 percent triggers a root cause analysis "
            "by the quality team; the rates of the quarter are listed below."),
          long_table(batches, [3 * cm, 3 * cm, 3 * cm, 4 * cm]),
          Spacer(1, 10),
          H(1, "3 Staffing"),
          P("Headcount reached 1,240 employees at the end of September.<super>1</super> The training programme on the "
            "LX-4 line was followed by 310 operators."),
          Paragraph("<super>1</super> Including 85 people on temporary contracts hired for the LX-4 launch.", NOTE),
          H(1, "4 Outlook"),
          P("The company expects its output to rise by six percent in the fourth quarter, once the second assembly "
            "line of Turin reaches its nominal speed.")]
    ctl.Doc(path, "Lumen Labs Quarterly Operations Review").build(s)
    return {"headings": [["1 Overview", 1], ["2 Production", 1], ["2.1 Output by site", 2],
                         ["2.2 Defect rates by batch", 2], ["3 Staffing", 1], ["4 Outlook", 1]],
            "n_figures": 1, "n_tables": 1,
            "needles": [{"query": "defect rate above 2.5 percent triggers a root cause analysis",
                         "section": "2.2 Defect rates by batch"},
                        {"query": "output to rise by six percent in the fourth quarter", "section": "4 Outlook"}]}


# ----------------------------------------------------------------------------- document 5
def concrete(path: Path) -> dict:
    lead = ("Temperature records. Temperatures must be logged every thirty minutes by sensors embedded in the "
            "concrete, from the end of the pour until the specified strength is reached. Any gap longer than two "
            "hours in the record invalidates the maturity estimate of the affected pour, which must then be checked "
            "with cores tested by an accredited laboratory before the formwork is removed.")
    s = [Paragraph("Concrete Curing and Strength Estimation Guide", TITLE)]
    s += [H(1, "1 Purpose"),
          P("This guide sets the rules used on our sites to decide when formwork can be removed. It relies on the "
            "maturity method, which estimates the strength of young concrete from its temperature history."),
          H(1, "2 Strength estimation"),
          H(2, "2.1 Maturity index"),
          P("The maturity index M is the sum, over the curing period, of the difference between the concrete "
            "temperature and a datum temperature, multiplied by the duration of each interval:"),
          formula(r"$M = \sum_{i} (T_i - T_0)\,\Delta t_i$"),
          P("The datum temperature T0 is minus ten degrees Celsius for ordinary Portland cement. The index is "
            "expressed in degree-hours, and a strength of 70 percent of the design strength is reached at about "
            "1,900 degree-hours for our standard mix."),
          H(2, "2.2 Monitoring"),
          H(3, lead),  # a heading style and a bookmark, as a run-in heading Docling took for a heading
          P("The sensor batteries last for 28 days, which covers the curing period of every mix used on our sites."),
          H(1, "3 Minimum curing durations"),
          P("When no sensor is installed, the formwork stays in place for at least the durations below."),
          ctl.table([["Cement type", "At 5 °C (days)", "At 20 °C (days)"], ["CEM I", "6", "3"], ["CEM II", "8", "4"],
                     ["CEM III", "12", "6"]], [5 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 10),
          H(1, "4 Acceptance"),
          P("A pour is accepted when all of the following conditions are met:"),
          ListFlowable([ListItem(Paragraph(t, BODY)) for t in
                        ["the maturity index has reached the target of the mix",
                         "no crack wider than 0.3 millimetres is visible",
                         "the site engineer has signed the pour record"]], bulletType="bullet")]
    ctl.Doc(path, "Concrete Curing and Strength Estimation Guide").build(s)
    return {"headings": [["1 Purpose", 1], ["2 Strength estimation", 1], ["2.1 Maturity index", 2],
                         ["2.2 Monitoring", 2], ["3 Minimum curing durations", 1], ["4 Acceptance", 1]],
            "n_figures": 0, "n_tables": 1, "n_formulas": 1, "false_headings": [lead[:40]],
            "needles": [{"query": "sensor batteries last for 28 days", "section": "2.2 Monitoring"},
                        {"query": "datum temperature minus ten degrees Celsius", "section": "2.1 Maturity index"}]}


# ----------------------------------------------------------------------------- document 6
def ferry(path: Path) -> dict:
    s = [Paragraph("Harbour Ferry Service Guide", TITLE)]
    s += [H(1, "Introduction"),
          P("The harbour ferry links the old port of Portbrise with the island of Ile Verte. The crossing takes "
            "twenty-five minutes and the service runs every day of the year, including public holidays."),
          H(1, "Timetable"),
          P("From April to October the first ferry leaves Portbrise at 6.40 and then every forty minutes until "
            "22.00. From November to March the interval is one hour and the last departure is at 20.30. "
            "Departures from Ile Verte take place twenty minutes after each arrival."),
          H(1, "Fares"),
          P("Tickets are sold at the harbour office and on the mobile application. Children under six travel free."),
          ctl.table([["Ticket", "Price"], ["Single", "4.50 EUR"], ["Return", "8.00 EUR"], ["Ten crossings", "36.00 EUR"],
                     ["Bicycle", "2.00 EUR"]], [3.5 * cm, 3 * cm]),
          Spacer(1, 8),
          H(1, "Accessibility"),
          P("Both ferries are accessible to passengers with reduced mobility. On board you will find:"),
          ListFlowable([ListItem(Paragraph(t, BODY)) for t in
                        ["a ramp at the bow, used at both landings", "two wheelchair spaces on the lower deck",
                         "an induction loop at the information desk"]], bulletType="bullet"),
          P("Dogs travel free when they are kept on a lead."),
          H(1, "Conclusion"),
          P("Passengers are asked to arrive five minutes before departure. In strong wind the harbour master may "
            "cancel crossings; cancellations are announced on the application at least thirty minutes in advance.")]
    TwoColumns(path, "Harbour Ferry Service Guide").build(s)
    return {"headings": [["Introduction", 1], ["Timetable", 1], ["Fares", 1], ["Accessibility", 1], ["Conclusion", 1]],
            "n_figures": 0, "n_tables": 1,
            "needles": [{"query": "first ferry leaves Portbrise at 6.40", "section": "Timetable"},
                        {"query": "two wheelchair spaces on the lower deck", "section": "Accessibility"}]}


# ----------------------------------------------------------------------------- document 7
def water(path: Path) -> dict:
    s = [Paragraph("Rivermouth Water Quality Report 2026", TITLE)]
    s += [H(1, "I. Introduction"),
          P("This report presents the quality of the water of the Rivermouth estuary measured between March and "
            "September 2026 by the municipal environment department."),
          H(1, "II. Sampling"),
          H(2, "A. Sites"),
          P("Five sampling sites, S1 to S5, were visited every two weeks. Site S3 lies downstream of the outlet of "
            "the treatment plant and site S5 at the mouth of the estuary."),
          H(2, "B. Methods"),
          P("Nitrates were measured by ion chromatography and Escherichia coli by membrane filtration, within six "
            "hours of each sampling."),
          H(1, "III. Results"),
          H(2, "A. Nitrates"),
          P("Nitrate concentrations stayed below the drinking water limit of 50 milligrams per litre at every site. "
            "The highest values were recorded in May, after heavy rain on the farmland upstream."),
          ctl.table([["Site", "Mean (mg/L)", "Maximum (mg/L)"], ["S1", "12", "19"], ["S2", "15", "24"],
                     ["S3", "21", "38"], ["S4", "17", "27"], ["S5", "9", "14"]], [3 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 10),
          H(2, "B. Bacteria"),
          P("Escherichia coli exceeded the bathing water threshold twice at site S3, on 14 May and on 2 July. "
            "Both events followed a storm overflow of the sewer network."),
          H(1, "IV. Conclusions"),
          P("The department recommends a second storage basin at the treatment plant, so that storm water no "
            "longer reaches the estuary without treatment."),
          H(1, "References"),
          P("[1] Estuary Monitoring Network. Sampling protocol for coastal waters, third edition, 2024."),
          P("[2] Municipal Environment Department. Rivermouth water quality report 2025."),
          H(1, "Appendix A. Laboratory"),
          P("All analyses were made by the Coastal Analysis Laboratory, accredited under the number LAB-4471 for "
            "nitrates and microbiology.")]
    ctl.Doc(path, "Rivermouth Water Quality Report 2026").build(s)
    return {"headings": [["I. Introduction", 1], ["II. Sampling", 1], ["A. Sites", 2], ["B. Methods", 2],
                         ["III. Results", 1], ["A. Nitrates", 2], ["B. Bacteria", 2], ["IV. Conclusions", 1],
                         ["References", 1], ["Appendix A. Laboratory", 1]],
            "n_figures": 0, "n_tables": 1,
            "needles": [{"query": "exceeded the bathing water threshold twice at site S3", "section": "B. Bacteria"},
                        {"query": "accredited under the number LAB-4471", "section": "Appendix A. Laboratory"}]}


# ----------------------------------------------------------------------------- document 8
def turbine(path: Path) -> dict:
    steps = ["Stop the rotor with the service brake and engage the rotor lock.",
             "Switch the controller to maintenance mode with the command shown in section 3.2.",
             "Check every blade bolt with a calibrated torque wrench, at the torque given in Table 2.",
             "Record the readings in the maintenance log and release the rotor lock."]
    s = [Paragraph("WT-90 Wind Turbine Maintenance Manual", TITLE)]
    s += [H(1, "1 Safety"),
          P("No one may climb the tower when the wind speed at hub height exceeds 15 metres per second. Two "
            "technicians are always present, and each carries a fall arrest harness and a radio."),
          H(1, "2 Intervals"),
          P("A visual inspection takes place every six months and a full inspection every year. The gearbox oil is "
            "changed every three years, or earlier if the oil analysis shows particles above the limit."),
          H(1, "3 Annual inspection"),
          H(2, "3.1 Procedure"),
          P("The annual inspection follows the steps below, in this order:"),
          ListFlowable([ListItem(Paragraph(t, BODY)) for t in steps], bulletType="1"),
          H(2, "3.2 Controller commands"),
          P("The controller is operated from the console in the nacelle:"),
          Preformatted("wtctl mode maintenance --confirm\nwtctl status\nmode: maintenance\nrotor_lock: engaged",
                       CODE),
          H(1, "4 Reference values"),
          P("Table 2 lists the tightening torques of the main bolted joints."),
          ctl.table([["Joint", "Torque (N m)"], ["Blade bolts", "1,450"], ["Tower flange", "2,900"],
                     ["Yaw bearing", "980"]], [6 * cm, 4 * cm]),
          Paragraph("Table 2. Tightening torques.", CAPTION)]
    ctl.Doc(path, "WT-90 Wind Turbine Maintenance Manual").build(s)
    return {"headings": [["1 Safety", 1], ["2 Intervals", 1], ["3 Annual inspection", 1], ["3.1 Procedure", 2],
                         ["3.2 Controller commands", 2], ["4 Reference values", 1]],
            "n_figures": 0, "n_tables": 1,
            "needles": [{"query": "wind speed at hub height exceeds 15 metres per second", "section": "1 Safety"},
                        {"query": "gearbox oil is changed every three years", "section": "2 Intervals"}]}


DOCUMENTS = (("01_aurora_bank_annual_report.pdf", ctl.aurora), ("02_alpine_botany_field_guide.pdf", ctl.botany),
             ("03_network_operations_handbook.pdf", ctl.netops), ("04_lumen_labs_operations_review.pdf", lumen),
             ("05_concrete_curing_guide.pdf", concrete), ("06_harbour_ferry_guide.pdf", ferry),
             ("07_rivermouth_water_quality.pdf", water), ("08_wt90_maintenance_manual.pdf", turbine))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, build in DOCUMENTS:
        expected[name] = build(OUT / name)
        print("wrote", OUT / name)
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
