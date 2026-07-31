"""Assemble the findings report as a PDF.

    .venv/bin/python scripts/make_report.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FIG = Path("docs/figures")
OUT = "docs/M&A-Findings-Report.pdf"

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#e2e1dd")
TINT = colors.HexColor("#f2f6fc")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=23, leading=27, textColor=INK,
                            spaceAfter=4, alignment=0),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=11.5,
                          leading=16, textColor=INK2, spaceAfter=16),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14,
                        leading=18, textColor=INK, spaceBefore=16,
                        spaceAfter=7),
    "body": ParagraphStyle("b", fontName="Helvetica", fontSize=10.5,
                           leading=15.5, textColor=INK, spaceAfter=9,
                           alignment=TA_JUSTIFY),
    "cap": ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=8.5,
                          leading=12, textColor=INK2, spaceBefore=3,
                          spaceAfter=13),
    "pull": ParagraphStyle("p", fontName="Helvetica-Bold", fontSize=12,
                           leading=17, textColor=BLUE, spaceBefore=6,
                           spaceAfter=10, leftIndent=10),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def fig(name, caption, width=6.4):
    """Size from the file's pixel aspect and pass both dimensions to the
    constructor -- setting drawWidth/drawHeight after construction gets
    recomputed during wrap, which overflows the frame."""
    from PIL import Image as PILImage
    px_w, px_h = PILImage.open(FIG / name).size
    w = width * inch
    # KeepTogether so a figure never lands on one page with its caption on
    # the next.
    return [KeepTogether([Image(str(FIG / name), width=w,
                                height=w * px_h / px_w),
                          P(caption, "cap")])]


def box(rows, widths):
    t = Table(rows, colWidths=[w * inch for w in widths])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), TINT),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def build():
    doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                            leftMargin=0.95 * inch, rightMargin=0.95 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="Predicting Company Takeovers from Public Filings",
                            author="Alban Maurel")
    f = []

    f.append(P("Predicting Company Takeovers<br/>from Public Filings", "title"))
    f.append(P("What eleven million government filings reveal about which "
               "companies get bought — and how I checked it wasn't luck.",
               "sub"))

    f.append(P("The question", "h"))
    f.append(P(
        "Roughly three in every hundred public companies get bought each year. "
        "The question is whether you can tell in advance <i>which</i> three — "
        "using only documents anyone can download for free."))
    f.append(P(
        "The answer is yes, to a useful degree. Out of every hundred companies "
        "on the shortlist this system produces, about twenty-eight were "
        "acquired within a year. Picking at random gets you three. That is "
        "roughly nine times better than chance, and it holds up under every "
        "test I could design to break it."))
    f.append(P("Twenty-eight in a hundred, against three in a hundred "
               "by chance.", "pull"))
    f.append(P(
        "It is also wrong about seven times out of ten. This is a way to "
        "narrow seven thousand companies down to twenty-five worth reading "
        "about. It is not a crystal ball, and the rest of this report is "
        "mostly about the difference."))

    f.append(P("Where the data comes from", "h"))
    f.append(P(
        "Every public company in America files documents with the Securities "
        "and Exchange Commission, and all of them are free. I downloaded "
        "eleven and a half million filing records covering 2016 to today, then "
        "turned them into a week-by-week picture of fifteen thousand "
        "companies: what they own and owe, what their executives are buying "
        "and selling, who has taken a stake in them, and what they announce."))
    f.extend(fig("funnel.png", "Each stage of the pipeline. Note the scale is "
                 "compressed — eleven million filings condense into about "
                 "2,500 actual takeovers to learn from."))

    f.append(P("The main result", "h"))
    f.append(P(
        "Each week the system ranks every company and hands back a shortlist. "
        "The chart below shows how often that shortlist was right, measured on "
        "a period the system had never seen while it was being built."))
    f.extend(fig("hit_rate.png",
                 "Longer lists dilute: the top fifty is the sweet spot. "
                 "Measured on 2024 through mid-2025, data the model was never "
                 "trained on."))
    f.append(P(
        "A shorter list is not automatically better. The top ten is slightly "
        "worse than the top fifty, which tells me the very highest-ranked "
        "picks are still a little scrambled — a known weakness rather than a "
        "mystery."))

    f.append(P("Why the obvious approach fails", "h"))
    f.append(P(
        "My first attempt used a simple model, and its most confident "
        "predictions were its worst — the top hundred contained no takeovers "
        "at all. The reason turned out to be company size."))
    f.extend(fig("size_hump.png",
                 "Takeover rate rises with size, then falls. A simple model "
                 "can only learn 'bigger is likelier', so it filled its "
                 "shortlist with giants — the least likely group of all."))
    f.append(P(
        "Mid-sized companies get bought. Giants are too expensive and minnows "
        "are not worth the trouble. A simple model cannot express a "
        "rise-then-fall relationship, so it confidently ranked the wrong "
        "companies first. Switching to a model that can bend fixed it."))

    f.append(P("Making sure it isn't luck", "h"))
    f.append(P(
        "A number this good deserves suspicion, so I attacked it. The hardest "
        "test is the simplest: scramble the answers and rebuild the model. If "
        "it still appears to work on nonsense, then the machinery is broken "
        "rather than the finding real."))
    f.extend(fig("shuffle.png",
                 "Twenty rebuilds on deliberately scrambled answers. The best "
                 "of them reached 7.6 percent. The real model sits far above "
                 "all of them."))
    f.append(P(
        "It collapsed, exactly as it should. I also reran the whole thing five "
        "times with different random starting points, and separately tested it "
        "on four different years."))
    f.extend(fig("stability.png",
                 "Five reruns land between 20 and 24 percent. Four separate "
                 "years land between 32 and 43. Neither depends on a lucky "
                 "draw or a single favourable period.", 6.2))

    f.append(P("What actually gives a company away", "h"))
    f.append(P(
        "The model was left to find its own patterns, but I can ask which "
        "signals genuinely carry information once everything else is "
        "accounted for. The bar length below measures how confident I can be "
        "that a signal is real rather than coincidence; anything past two is "
        "unlikely to be chance."))
    f.extend(fig("signals.png",
                 "Nine signals that survive careful statistical testing. Most "
                 "are things a company does, or stops doing, rather than "
                 "anything it says about being for sale.", 6.7))
    f.append(P(
        "The strongest single warning sign is a competitor being bought. "
        "Takeovers arrive in waves through an industry, and once one company "
        "in a sector goes, its rivals become targets."))
    f.append(P(
        "The most interesting one is quieter. Company insiders — executives "
        "and directors — trade their own shares on a fairly regular rhythm. "
        "When a deal is being negotiated they are barred from trading, so "
        "that rhythm simply stops. <b>Nobody announces a silence.</b> A "
        "company can control every word of its public messaging and still "
        "give itself away by what stops happening."))

    f.append(P("Does it beat what people already knew?", "h"))
    f.append(P(
        "Researchers have known since the 1980s that takeover targets tend to "
        "be small, cheap and cash-rich. If my system merely rediscovered that, "
        "it would be worth little. So I built it twice."))
    f.extend(fig("nested.png",
                 "Adding the new filing-based signals lifts the hit rate by "
                 "roughly a fifth over the long-established indicators alone.",
                 5.4))

    f.append(P("Four mistakes I caught", "h"))
    f.append(P(
        "Each of these produced a confident, wrong answer before I found it. "
        "They are the part of this project I would most want a reader to "
        "notice."))
    f.append(box([
        ["What went wrong", "How it was caught"],
        [Paragraph("<b>Counting the unfinished.</b> Companies in the final "
                   "year of data hadn't had time to be acquired yet, so they "
                   "were all filed as failures.", S["body"]),
         Paragraph("Suspicious that year-by-year tests scored far higher than "
                   "the overall test. Fixing it moved the result from 19.6 to "
                   "27.5 percent.", S["body"])],
        [Paragraph("<b>Shell companies.</b> Blank-cheque vehicles exist purely "
                   "to merge, so the model partly learned to spot those "
                   "instead of real businesses.", S["body"]),
         Paragraph("Noticed nine of the top fifteen names were shells. "
                   "Removing all 1,707 of them costs some accuracy but the "
                   "finding survives.", S["body"])],
        [Paragraph("<b>A measurement that measured nothing.</b> My 'top 100' "
                   "score turned out to cover just three companies, one "
                   "appearing 84 weeks running.", S["body"]),
         Paragraph("Checked how many distinct companies were actually in the "
                   "list. Replaced it with a weekly shortlist, which is how "
                   "anyone would really use it.", S["body"])],
        [Paragraph("<b>Late paperwork.</b> The document I used to date each "
                   "takeover is filed one to two months <i>after</i> the deal "
                   "becomes public.", S["body"]),
         Paragraph("Blanked out the final twelve weeks before each deal and "
                   "reran. Performance held, so the model predicts rather "
                   "than reads the news.", S["body"])],
    ], [2.75, 3.55]))
    f.append(Spacer(1, 14))
    f.extend(fig("mistakes.png",
                 "Two of the four corrections, and what each did to the "
                 "headline number.", 5.8))

    f.append(P("What this cannot do", "h"))
    f.append(P(
        "<b>It is wrong most of the time.</b> Seven or eight of every ten "
        "names on the shortlist are not acquired. This is a filter, not a "
        "forecast."))
    f.append(P(
        "<b>It predicts a year, not a date.</b> The system says a deal is "
        "likely within twelve months. It says nothing about when."))
    f.append(P(
        "<b>It cannot see share prices.</b> No free source keeps price history "
        "for companies that have been acquired — precisely the ones that "
        "matter here. So the system knows nothing about whether a share price "
        "has been falling, which is one of the better-known warning signs."))
    f.append(P(
        "<b>It misses hostile takeovers.</b> The records I used to define a "
        "takeover cover negotiated deals well and hostile bids poorly. Tested "
        "against hostile deals it had never seen, it still beat chance by "
        "three and a half times — real, but much weaker."))

    f.append(P("What comes next", "h"))
    f.append(P(
        "Access to a university share-price archive would close the largest "
        "gap: it supplies price history for acquired companies, a cleaner "
        "record of every takeover including hostile ones, and — most "
        "importantly — a way to ask the question I actually care about. "
        "<b>Does a company's paperwork betray a deal before the stock market "
        "notices?</b> That question is currently unanswerable, and it is the "
        "more interesting one."))

    doc.build(f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
