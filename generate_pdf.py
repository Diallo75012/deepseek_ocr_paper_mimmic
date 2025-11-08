# create_pdf.py
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
import random

doc = SimpleDocTemplate("manga_kissa_report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>Manga Kissa Shibuya — Daily Activity Report</b>", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "This report summarizes the usage of the Manga Kissa lounge in Shibuya. "
    "It includes visitor attendance, reservation summary, and snack sales statistics for the last 24 hours.",
    styles["Normal"]))

story.append(Spacer(1, 20))

# Attendance Table
data = [["Time Slot", "Visitors", "Reserved Seats"],
        ["00:00–06:00", "42", "18"],
        ["06:00–12:00", "58", "21"],
        ["12:00–18:00", "74", "30"],
        ["18:00–24:00", "96", "42"]]
t = Table(data, colWidths=[120, 100, 120])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("ALIGN", (1,1), (-1,-1), "CENTER")
]))
story.append(Paragraph("<b>Table 1 — Attendance by Time Slot</b>", styles["Heading3"]))
story.append(t)

story.append(Spacer(1, 20))

# Simple bar chart for snack sales
drawing = Drawing(400, 200)
chart = VerticalBarChart()
chart.x = 30
chart.y = 30
chart.height = 120
chart.width = 300
chart.data = [[random.randint(30,100) for _ in range(5)]]
chart.categoryAxis.categoryNames = ["Coffee", "Ramen", "Manga Set", "Dessert", "Soft Drinks"]
chart.bars[0].fillColor = colors.lightblue
drawing.add(chart)
drawing.add(String(100, 170, "Snack Sales (Units)", fontSize=12))
story.append(Paragraph("<b>Figure 1 — Snack Sales Statistics</b>", styles["Heading3"]))
story.append(drawing)

story.append(Spacer(1, 20))
story.append(Paragraph(
    "Observation: evening hours (18:00–24:00) remain the busiest. Snack sales "
    "correlate with total attendance. Reservation system performed within acceptable limits.",
    styles["Normal"]))

story.append(Spacer(1, 12))
story.append(Paragraph("Prepared by: AI Ops Assistant · Date: Today", styles["Italic"]))

doc.build(story)
print("Generated manga_kissa_report.pdf ✅")
