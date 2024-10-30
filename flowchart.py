from graphviz import Digraph
from fpdf import FPDF

# Create the flowchart using Graphviz
dot = Digraph()

# Main components
dot.node("A", "Sprite: Avery")
dot.node("B", "Costumes")
dot.node("C", "Sounds")
dot.node("D", "Scripts")

# Connect components
dot.edge("A", "B")
dot.edge("A", "C")
dot.edge("A", "D")

# Costumes component
dot.node("B1", "Walking")
dot.node("B2", "Sitting")
dot.node("B3", "Running")
dot.edge("B", "B1")
dot.edge("B", "B2")
dot.edge("B", "B3")

# Scripts component
dot.node("D1", "Motion Blocks")
dot.node("D2", "Looks Blocks")
dot.node("D3", "Sounds Blocks")
dot.node("D4", "Control Blocks")
dot.node("D5", "Events Blocks")
dot.node("D6", "Sensing Blocks")
dot.node("D7", "Operators Blocks")
dot.node("D8", "Variables Blocks")
dot.node("D9", "My Blocks")

dot.edge("D", "D1")
dot.edge("D", "D2")
dot.edge("D", "D3")
dot.edge("D", "D4")
dot.edge("D", "D5")
dot.edge("D", "D6")
dot.edge("D", "D7")
dot.edge("D", "D8")
dot.edge("D", "D9")

# Save flowchart as PDF
dot.format = "png"
flowchart_path = "/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/flowchart.png"
dot.render(flowchart_path, view=False)

# Create PDF
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Scratch Sprite Components Flowchart", ln=True, align="C")

# Add image
pdf.image(flowchart_path, x=10, y=30, w=180)

# Save PDF
pdf_output_path = "/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/scratch_flowchart.pdf"
pdf.output(pdf_output_path)

pdf_output_path