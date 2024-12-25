import re
from collections import defaultdict
from docx import Document
import csv

# Function to parse node counts from text
def parse_nodes(text):
    node_pattern = r"\\((\\d+)\\)"  # Matches numbers inside parentheses
    return sum(int(match) for match in re.findall(node_pattern, text))

# Function to process the document
def process_docx(file_path):
    doc = Document(file_path)
    node_data = defaultdict(int)
    current_file = None

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        # Check for new file identifier based on "S/N" column
        if text.startswith("S/N") or text.isdigit():
            current_file = text
        elif current_file:
            # Add node counts for the current file
            node_data[current_file] += parse_nodes(text)

    return node_data

# Function to process the CSV file
def process_csv(file_path):
    node_data = defaultdict(int)
    current_file = None

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].startswith("S/N"):
                current_file = row[0]
            elif current_file:
                node_data[current_file] += parse_nodes(" ".join(row))

    return node_data

# Function to generate a summary
def generate_summary(node_data):
    summary = "Summary of Node Counts:\n\n"
    total_files = len(node_data)
    total_nodes = sum(node_data.values())

    max_file = max(node_data, key=node_data.get)
    min_file = min(node_data, key=node_data.get)

    summary += f"Total files: {total_files}\n"
    summary += f"Total nodes: {total_nodes}\n"
    summary += f"File with the most nodes: {max_file} ({node_data[max_file]} nodes)\n"
    summary += f"File with the fewest nodes: {min_file} ({node_data[min_file]} nodes)\n\n"
    summary += "Node counts per file:\n"

    for file, count in node_data.items():
        summary += f"{file}: {count} nodes\n"

    return summary

# Main function to execute the script
def main():
    docx_path = "/Users/samueliwuchukwu/downloads/UnderstandingScratchRevisonsTagging2.docx"  # Update with your docx path
    csv_path = "/Users/samueliwuchukwu/downloads/UnderTag.csv"  # Update with your CSV path

    # Process files
    docx_data = process_docx(docx_path)
    csv_data = process_csv(csv_path)

    # Combine results
    combined_data = defaultdict(int, docx_data)
    for file, count in csv_data.items():
        combined_data[file] += count

    # Generate and print summary
    summary = generate_summary(combined_data)
    print(summary)

if __name__ == "__main__":
    main()
