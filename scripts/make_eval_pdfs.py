"""Generate the synthetic part of the evaluation set: 9 fictitious PDFs, each built on a known weak point.

Every document is invented and generated here, so the set can ship with the kit (no licence issue, unlike Open RAG
Bench). The arXiv paper 2404.02228 (headings run into their paragraph, with PDF bookmarks) completes the set; it is
downloaded by ``scripts/fetch_eval_arxiv.py``.

* 01 journal article: headings run into their paragraph ("2.1. Title. Text..."), with PDF bookmarks; a "Remark 1."
  lead-in that is not a heading;
* 02 cooperative bulletin: the same run-in headings, without bookmarks (known limit: nothing marks them);
* 03 command-line reference: code blocks of short lines (Docling cuts them into lines of a few words);
* 04 solar farm report: values printed on the charts only;
* 05 fleet register: a 60-row table over two pages, a question computed from two rows;
* 06 bird newsletter: two columns, footnotes;
* 07 signalling standard: three heading levels, titled appendices, references (not indexed);
* 08 heat exchanger note: formulas drawn as images;
* 09 remote work policy: unnumbered headings set in the body size (bold), lists, an answer across two sections.

``expected.json`` gives the real headings of each PDF; the questions are in ``eval_dataset.jsonl`` (written by hand;
``tests/test_eval_set.py`` checks that every evidence string is in its PDF).

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
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Frame, Image, ListFlowable, ListItem, PageTemplate, Paragraph, Preformatted,  # noqa: E402
                                Spacer, Table, TableStyle)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "pdf_eval"
H, P, BODY, CAPTION, TITLE = ctl.H, ctl.P, ctl.BODY, ctl.CAPTION, ctl.TITLE
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=8.5, leading=11)
CODE = ParagraphStyle("CodeBlock", fontName="Courier", fontSize=9, leading=12, leftIndent=12)
AUTHORS = ParagraphStyle("Authors", parent=BODY, alignment=1, fontName="Times-Italic")
BOLD_BODY = ParagraphStyle("BoldBody", parent=BODY, fontName="Times-Bold", spaceBefore=6, spaceAfter=4)


class EvalDoc(ctl.Doc):
    """``ctl.Doc`` (a bookmark per heading) plus the bookmarks of run-in headings; ``bookmarks=False`` writes none."""

    def __init__(self, path, title, *, bookmarks: bool = True, columns: int = 1):
        super().__init__(path, title)
        self.bookmarks = bookmarks
        if columns == 2:
            gap, w = 0.8 * cm, (self.width - 0.8 * cm) / 2
            frames = [Frame(self.leftMargin, self.bottomMargin, w, self.height, id="left"),
                      Frame(self.leftMargin + w + gap, self.bottomMargin, w, self.height, id="right")]
            self.pageTemplates = []
            self.addPageTemplates([PageTemplate(id="two", frames=frames, onPage=self._footer)])

    def afterFlowable(self, fl):
        if not self.bookmarks:
            return
        run_in = getattr(fl, "_run_in", None)
        if run_in:
            level, title = run_in
            key = f"r{self._n}"
            self._n += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=level - 1, closed=False)
            return
        super().afterFlowable(fl)


def run_in(level: int, number: str, title: str, text: str) -> Paragraph:
    """A heading set at the start of its paragraph, as in many journals; its bookmark keeps the title only."""
    p = Paragraph(f"<b>{number} {title}.</b> {text}", BODY)
    p._run_in = (level, title)
    return p


def bold_heading(text: str) -> Paragraph:
    """A heading in the size of the body text, only bold: no size tells it apart."""
    p = Paragraph(text, BOLD_BODY)
    p._hlevel = 1
    return p


def bullets(items, kind: str = "bullet") -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(t, BODY)) for t in items], bulletType=kind)


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


def line_chart(title: str, labels, values, ylabel: str) -> Image:
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=140)
    ax.plot(labels, values, marker="o", color="#2F8F5B")
    for x, y in zip(labels, values):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(alpha=.3)
    fig.tight_layout()
    return _image(fig, 12, 2.8 / 5.2)


def formula(tex: str) -> Image:
    fig = plt.figure(figsize=(4.4, 0.7), dpi=200)
    fig.text(0.5, 0.5, tex, fontsize=16, ha="center", va="center")
    return _image(fig, 9, 0.7 / 4.4)


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


def _filler(topic: str) -> str:
    return (f"The working group reviewed the {topic} at each of its monthly meetings and recorded every decision in "
            "the minutes, so that the reasoning behind each choice can be traced by later teams.")


# ----------------------------------------------------------------------------- document 1
def hydrology(path: Path) -> dict:
    s = [Paragraph("Groundwater Recharge in the Talvera Basin", TITLE),
         Paragraph("M. Okafor and L. Brandt, Institute of Fictional Hydrology", AUTHORS),
         run_in(1, "1.", "Introduction", "The Talvera basin supplies drinking water to about 180,000 inhabitants. "
                "Its aquifer is recharged mainly in winter, and the dry summers of the last decade raised the question "
                "of how much water can be pumped without lowering the water table. " + _filler("study design")),
         run_in(1, "2.", "Study area and data", "The basin covers 412 square kilometres between the Serra hills and "
                "the river Talvera. " + _filler("data sources")),
         run_in(2, "2.1.", "Monitoring wells", "Water levels were recorded every hour in 37 monitoring wells from "
                "January 2019 to December 2025. Six wells with gaps longer than three months were excluded, which "
                "left 31 wells for the analysis."),
         run_in(2, "2.2.", "Rainfall records", "Daily rainfall comes from the eight stations of the regional weather "
                "service. The mean annual rainfall over the period was 640 millimetres, with a minimum of 410 "
                "millimetres in 2022."),
         run_in(1, "3.", "Recharge estimation", "Recharge was estimated with the water table fluctuation method, "
                "which multiplies the rise of the water table after each rain event by the specific yield of the "
                "aquifer. " + _filler("estimation method")),
         Paragraph("<b>Remark 1.</b> The specific yield was set to 0.08 for the sandy layers and 0.03 for the clay "
                   "layers, the values measured in the pumping tests of 2018.", BODY),
         run_in(2, "3.1.", "Uncertainty", "A Monte Carlo analysis with 5,000 draws of the specific yield gave a 90 "
                "percent interval of 52 to 81 millimetres per year for the mean recharge."),
         run_in(1, "4.", "Results", "The mean annual recharge was 66 millimetres, that is about 27 million cubic "
                "metres per year over the basin. Recharge fell to 19 millimetres in the dry year 2022."),
         run_in(1, "5.", "Discussion", "Pumping currently takes 24 million cubic metres per year, close to the mean "
                "recharge. The authors recommend capping the pumping at 20 million cubic metres in dry years and "
                "building two infiltration basins near the river."),
         H(1, "References"),
         P("[1] Healy, R. Estimating groundwater recharge. Fictional Press, 2010.")]
    EvalDoc(path, "Groundwater Recharge in the Talvera Basin").build(s)
    return {"headings": [["1. Introduction", 1], ["2. Study area and data", 1], ["2.1. Monitoring wells", 2],
                         ["2.2. Rainfall records", 2], ["3. Recharge estimation", 1], ["3.1. Uncertainty", 2],
                         ["4. Results", 1], ["5. Discussion", 1], ["References", 1]],
            "n_figures": 0, "n_tables": 0, "run_in_headings": True, "bookmarks": True,
            "needles": [{"query": "31 wells for the analysis", "section": "2.1. Monitoring wells"},
                        {"query": "capping the pumping at 20 million cubic metres", "section": "5. Discussion"}]}


# ----------------------------------------------------------------------------- document 2
def vineyard(path: Path) -> dict:
    s = [Paragraph("Valmora Wine Cooperative Bulletin, Harvest 2026", TITLE),
         run_in(1, "1.", "Harvest overview", "The 2026 harvest started on 4 September and ended on 2 October. "
                "The 214 member growers delivered 9,850 tonnes of grapes, eleven percent more than in 2025."),
         run_in(1, "2.", "Grape quality", "The average sugar content reached 22.4 degrees Brix. " + _filler("quality "
                "criteria")),
         run_in(2, "2.1.", "Hail damage", "A hailstorm on 18 July hit the plots of the northern slope; 37 growers "
                "declared losses, covered by the mutual fund of the cooperative."),
         run_in(1, "3.", "Payments", "The advance paid at delivery is 0.62 euros per kilogram. The balance will be "
                "paid in March, once the wines of the vintage have been valued."),
         run_in(1, "4.", "Next general assembly", "The general assembly will take place on 12 December in the "
                "cellar hall; members who cannot attend may give their proxy to another member.")]
    EvalDoc(path, "Valmora Wine Cooperative Bulletin", bookmarks=False).build(s)
    return {"headings": [["1. Harvest overview", 1], ["2. Grape quality", 1], ["2.1. Hail damage", 2],
                         ["3. Payments", 1], ["4. Next general assembly", 1]],
            "n_figures": 0, "n_tables": 0, "run_in_headings": True, "bookmarks": False,
            "needles": [{"query": "9,850 tonnes of grapes", "section": "1. Harvest overview"},
                        {"query": "advance paid at delivery is 0.62 euros per kilogram", "section": "3. Payments"}]}


# ----------------------------------------------------------------------------- document 3
def orbit_cli(path: Path) -> dict:
    s = [Paragraph("Orbit Backup Tool: Command Reference", TITLE)]
    s += [H(1, "1 Installation"),
          P("Orbit runs on Linux and Windows. After installation, register the machine with the backup server:"),
          Preformatted("orbit register --server backup.example.org\norbit --version\norbit 4.2.1", CODE),
          H(1, "2 Daily use"),
          H(2, "2.1 Starting a backup"),
          P("A backup of the default profile starts with the command below; it prints a job number."),
          Preformatted("orbit backup start --profile default\njob: 5521\nstate: queued", CODE),
          H(2, "2.2 Restoring files"),
          P("To restore a single folder from the last backup into a separate directory, run:"),
          Preformatted("orbit restore --path /srv/data --target /tmp/restore\nrestored: 1,204 files", CODE),
          H(1, "3 Retention"),
          P("Daily backups are kept for 30 days, weekly backups for 26 weeks and monthly backups for 7 years. The "
            "retention can be shortened per profile but never extended beyond these limits."),
          H(1, "4 Troubleshooting"),
          P("If a backup stays queued for more than one hour, check the connection to the server:"),
          Preformatted("orbit doctor --network\nserver: reachable\nlatency: 18 ms", CODE)]
    EvalDoc(path, "Orbit Backup Tool: Command Reference").build(s)
    return {"headings": [["1 Installation", 1], ["2 Daily use", 1], ["2.1 Starting a backup", 2],
                         ["2.2 Restoring files", 2], ["3 Retention", 1], ["4 Troubleshooting", 1]],
            "n_figures": 0, "n_tables": 0, "code_blocks": 4,
            "needles": [{"query": "weekly backups for 26 weeks", "section": "3 Retention"},
                        {"query": "orbit doctor --network", "section": "4 Troubleshooting"}]}


# ----------------------------------------------------------------------------- document 4
def solar(path: Path) -> dict:
    months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    s = [Paragraph("Cardona Solar Farm: Performance Report, Summer 2026", TITLE)]
    s += [H(1, "1 Summary"),
          P("The Cardona solar farm has an installed capacity of 48 megawatts peak. The summer was sunnier than the "
            "ten-year average, but a transformer failure stopped one of the four blocks for eleven days."),
          H(1, "2 Production"),
          P("The monthly energy delivered to the grid is shown in the chart below."),
          ctl.fig_block(bar_chart("Energy delivered to the grid", months, [6.1, 7.4, 8.2, 8.9, 7.0, 5.6], "GWh"),
                        "Figure 1. Monthly energy delivered, in gigawatt-hours."),
          H(1, "3 Availability"),
          P("Availability is the share of the time during which the inverters were able to produce."),
          ctl.fig_block(line_chart("Availability", months, [99.1, 98.7, 99.3, 91.2, 98.9, 99.4], "%"),
                        "Figure 2. Monthly availability of the inverters, in percent."),
          P("The drop of one month is due to the transformer failure of block C, repaired with a spare unit."),
          H(1, "4 Maintenance"),
          P("The panels were cleaned twice, in May and in August. Vegetation under the rows was grazed by a flock of "
            "320 sheep from a neighbouring farm instead of being mowed.")]
    EvalDoc(path, "Cardona Solar Farm Performance Report").build(s)
    return {"headings": [["1 Summary", 1], ["2 Production", 1], ["3 Availability", 1], ["4 Maintenance", 1]],
            "n_figures": 2, "n_tables": 0,
            "chart_values": {"energy_gwh": dict(zip(months, [6.1, 7.4, 8.2, 8.9, 7.0, 5.6])),
                             "availability_pct": dict(zip(months, [99.1, 98.7, 99.3, 91.2, 98.9, 99.4]))},
            "needles": [{"query": "flock of 320 sheep", "section": "4 Maintenance"},
                        {"query": "installed capacity of 48 megawatts peak", "section": "1 Summary"}]}


# ----------------------------------------------------------------------------- document 5
FLEET_DEPOTS = ("Arles", "Nimes", "Sete")


def fleet_rows() -> list[list[str]]:
    rows = [["Vehicle", "Model", "Depot", "Fuel (L/100 km)"]]
    for i in range(60):
        plate = f"FV-{301 + i}"
        model = ("Kestrel van", "Atlas truck", "Comet car")[i % 3]
        depot = FLEET_DEPOTS[(i // 3) % 3]
        fuel = {"Kestrel van": 8.0, "Atlas truck": 21.0, "Comet car": 5.5}[model] + (i * 7 % 10) / 10
        if plate == "FV-352":
            fuel = 27.4  # the outlier of the register
        rows.append([plate, model, depot, f"{fuel:.1f}"])
    return rows


def fleet(path: Path) -> dict:
    s = [Paragraph("Municipal Fleet Fuel Register 2026", TITLE)]
    s += [H(1, "1 Scope"),
          P("This register lists the 60 vehicles of the municipal fleet and their average fuel consumption over the "
            "first half of 2026. A vehicle above 25 litres per 100 kilometres is sent to the workshop for a check."),
          H(1, "2 Register"),
          long_table(fleet_rows(), [3 * cm, 3.5 * cm, 3 * cm, 4 * cm]),
          Spacer(1, 10),
          H(1, "3 Replacement plan"),
          P("The Comet cars will be replaced by electric cars from 2027, starting with the depot of Sete, which has "
            "the shortest daily routes.")]
    EvalDoc(path, "Municipal Fleet Fuel Register 2026").build(s)
    return {"headings": [["1 Scope", 1], ["2 Register", 1], ["3 Replacement plan", 1]],
            "n_figures": 0, "n_tables": 1,
            "needles": [{"query": "above 25 litres per 100 kilometres is sent to the workshop", "section": "1 Scope"}]}


# ----------------------------------------------------------------------------- document 6
def birds(path: Path) -> dict:
    s = [Paragraph("Coastal Birds Newsletter, Autumn 2026", TITLE)]
    s += [H(1, "Spoonbill colony"),
          P("The spoonbill colony of the Brena marshes raised 146 chicks this year, a record since the first count in "
            "2009.<super>1</super> The colony now nests on the two islets created in 2021."),
          Paragraph("<super>1</super> Counts made from the observation tower by the volunteers of the reserve.", NOTE),
          H(1, "Ringing campaign"),
          P("From August to October, 2,318 birds of 41 species were ringed at the Cape Lumo station. The most "
            "ringed species was the sedge warbler, with 612 birds.<super>2</super>"),
          Paragraph("<super>2</super> One sedge warbler ringed at Cape Lumo was caught again in Senegal 19 days "
                    "later.", NOTE),
          H(1, "Volunteer days"),
          P("Volunteers meet on the first Saturday of each month to clear the reed beds. Boots and gloves are lent "
            "by the reserve; bring a packed lunch."),
          bullets(["meeting point: the car park of the Brena visitor centre", "start: 9.00, end: 15.30",
                   "registration: by e-mail one week before"]),
          H(1, "Agenda"),
          P("The annual seabird count will take place on 15 November. The talk on migratory routes by Dr. Inès "
            "Varga is moved to 6 December because of the renovation of the conference room.")]
    EvalDoc(path, "Coastal Birds Newsletter", columns=2).build(s)
    return {"headings": [["Spoonbill colony", 1], ["Ringing campaign", 1], ["Volunteer days", 1], ["Agenda", 1]],
            "n_figures": 0, "n_tables": 0, "two_columns": True,
            "needles": [{"query": "146 chicks this year", "section": "Spoonbill colony"},
                        {"query": "caught again in Senegal 19 days later", "section": "Ringing campaign"}]}


# ----------------------------------------------------------------------------- document 7
def signalling(path: Path) -> dict:
    s = [Paragraph("Signalling Standard SR-12: Level Crossings", TITLE)]
    s += [H(1, "1 General"),
          P("This standard applies to the automatic level crossings of the regional network. It replaces standard "
            "SR-09 from 1 January 2027."),
          H(1, "2 Equipment"),
          H(2, "2.1 Barriers"),
          H(3, "2.1.1 Closing time"),
          P("The barriers must be fully lowered at least 12 seconds before the train reaches the crossing, and the "
            "lowering itself must last between 6 and 10 seconds."),
          H(3, "2.1.2 Lighting"),
          P("Each barrier arm carries three red LED lamps, the lamp at the tip flashing at 60 cycles per minute."),
          H(2, "2.2 Detection"),
          H(3, "2.2.1 Approach sensors"),
          P("Two axle counters are placed on each track, the first one at a distance that gives 35 seconds of warning "
            "at the line speed."),
          H(3, "2.2.2 Obstacle detection"),
          P("Crossings on lines faster than 120 kilometres per hour are fitted with a radar that detects a vehicle "
            "trapped between the barriers and stops the train with the signal protecting the crossing."),
          H(1, "3 Inspection"),
          P("Every crossing is inspected every four weeks. The inspection report is filed within two working days."),
          H(1, "References"),
          P("[1] Regional Rail Authority. Standard SR-09, Automatic level crossings, 2015."),
          H(1, "Appendix A. Test equipment"),
          P("The closing time is measured with a calibrated stopwatch of class B or better, from the first flash of "
            "the lamps to the end of the lowering."),
          H(1, "Appendix B: Glossary"),
          P("Axle counter: a device that counts the axles of a train entering and leaving a section of track.")]
    EvalDoc(path, "Signalling Standard SR-12").build(s)
    return {"headings": [["1 General", 1], ["2 Equipment", 1], ["2.1 Barriers", 2], ["2.1.1 Closing time", 3],
                         ["2.1.2 Lighting", 3], ["2.2 Detection", 2], ["2.2.1 Approach sensors", 3],
                         ["2.2.2 Obstacle detection", 3], ["3 Inspection", 1], ["References", 1],
                         ["Appendix A. Test equipment", 1], ["Appendix B: Glossary", 1]],
            "n_figures": 0, "n_tables": 0,
            "needles": [{"query": "fully lowered at least 12 seconds before the train", "section": "2.1.1 Closing time"},
                        {"query": "calibrated stopwatch of class B", "section": "Appendix A. Test equipment"}]}


# ----------------------------------------------------------------------------- document 8
def exchanger(path: Path) -> dict:
    s = [Paragraph("Design Note DN-7: Plate Heat Exchanger for the District Heating Loop", TITLE)]
    s += [H(1, "1 Duty"),
          P("The exchanger transfers the heat of the primary loop to the secondary loop of the Riverside district. "
            "The design duty is 2.4 megawatts, with the primary water entering at 95 degrees Celsius."),
          H(1, "2 Sizing method"),
          H(2, "2.1 Logarithmic mean temperature difference"),
          P("The driving temperature difference is the logarithmic mean of the differences at both ends:"),
          formula(r"$\Delta T_{lm} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}$"),
          P("With 25 kelvin at the hot end and 15 kelvin at the cold end, the logarithmic mean is 19.6 kelvin."),
          H(2, "2.2 Required area"),
          P("The heat transfer area follows from the duty, the overall heat transfer coefficient U and the mean "
            "temperature difference:"),
          formula(r"$A = \frac{Q}{U \, \Delta T_{lm}}$"),
          P("With U equal to 4,500 watts per square metre and kelvin, the required area is 27.2 square metres; the "
            "selected unit has 31 square metres, which leaves a margin for fouling."),
          H(1, "3 Materials"),
          P("The plates are made of stainless steel 316L, 0.5 millimetres thick, and the gaskets of EPDM rated for "
            "130 degrees Celsius.")]
    EvalDoc(path, "Design Note DN-7").build(s)
    return {"headings": [["1 Duty", 1], ["2 Sizing method", 1], ["2.1 Logarithmic mean temperature difference", 2],
                         ["2.2 Required area", 2], ["3 Materials", 1]],
            "n_figures": 0, "n_tables": 0, "n_formulas": 2,
            "needles": [{"query": "required area is 27.2 square metres", "section": "2.2 Required area"},
                        {"query": "stainless steel 316L", "section": "3 Materials"}]}


# ----------------------------------------------------------------------------- document 9
def remote_work(path: Path) -> dict:
    s = [Paragraph("Remote Work Policy", TITLE)]
    s += [bold_heading("Purpose"),
          P("This policy sets the rules for working from home at Northwind Logistics. It applies to all employees "
            "whose duties can be carried out away from the warehouses."),
          bold_heading("Eligibility"),
          P("Employees become eligible after a probation period of three months. Team leaders decide on each request "
            "within ten working days."),
          bold_heading("Working days"),
          P("Eligible employees may work from home up to two days per week. Tuesday is a common office day for all "
            "teams and cannot be taken as a remote day."),
          bold_heading("Equipment"),
          P("The company provides a laptop, a headset and a second screen. A monthly allowance of 30 euros covers "
            "internet and electricity costs."),
          bullets(["equipment stays the property of the company", "faults are reported to the IT desk",
                   "a docking station can be borrowed on request"]),
          bold_heading("Security"),
          P("Work from public places such as cafes or trains is not allowed for tasks that involve customer data. "
            "The VPN must be used for every connection to the company systems.")]
    EvalDoc(path, "Remote Work Policy").build(s)
    return {"headings": [["Purpose", 1], ["Eligibility", 1], ["Working days", 1], ["Equipment", 1], ["Security", 1]],
            "n_figures": 0, "n_tables": 0, "body_size_headings": True,
            "needles": [{"query": "monthly allowance of 30 euros", "section": "Equipment"},
                        {"query": "Tuesday is a common office day", "section": "Working days"}]}


DOCUMENTS = (("01_talvera_groundwater_article.pdf", hydrology), ("02_valmora_cooperative_bulletin.pdf", vineyard),
             ("03_orbit_backup_cli_reference.pdf", orbit_cli), ("04_cardona_solar_report.pdf", solar),
             ("05_municipal_fleet_register.pdf", fleet), ("06_coastal_birds_newsletter.pdf", birds),
             ("07_sr12_level_crossing_standard.pdf", signalling), ("08_dn7_heat_exchanger_note.pdf", exchanger),
             ("09_northwind_remote_work_policy.pdf", remote_work))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, build in DOCUMENTS:
        expected[name] = build(OUT / name)
        print("wrote", OUT / name)
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
