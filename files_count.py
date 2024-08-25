
from PyPDF2 import PdfReader
import csv
import os
import re
import fitz
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
        
        matched_pattern =  re.compile(r'\s*- Script\d+ contains nodes such as\s+(?:\s*-\s+.+\s*\(.+\)\s*)+', re.IGNORECASE)
        
        node_pattern = re.compile(r'- (.*?) \((\d+)\)')
        matched_text = re.findall(matched_pattern,text)
        print(matched_text)
        #print(type(matched_text))
        #print(len(matched_text))
        for match in matched_text:
            print(match)
            #print(type(match))
            #print("actual node",match[1])
            nodes_match = re.findall(node_pattern,match)
            print(nodes_match)
        #matched_final = re.findall([matched_pattern,node_pattern],text)
        #print(matched_final)
    #print(text)
    return text
# Function to clean and format the extracted text

def extract_match_file(txt_path,output_file):
   node_counts = defaultdict(int)
   with open(txt_path, 'r') as file, open(output_file, 'w') as out_file:
        for line in file:
            # Match lines that contain nodes
            node_pattern = re.compile(r'\s*\* (.*?) \((\d+)\)')
            node_match = re.match(node_pattern, line)
            if node_match:
                node_text = node_match.group(1).strip()
                count = int(node_match.group(2))
                node_text_lower = node_text.strip().lower()
                if node_text_lower not in node_counts:
                    
                    node_counts[node_text_lower] = count
                node_counts[node_text_lower] += count
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Node_Type', 'Count'])
            for node_type, count in node_counts.items():
                csv_writer.writerow([node_type, count])    
        
   
def clean_extracted_text(text):
    # Replacing multiple newlines and spaces with a single newline
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to extract nodes and counts from the text
def extract_nodes(text):
    # Regular expression to match sections starting with '-Script' and extract nodes
    #script_pattern = re.compile(r'\s*- Script\d+ contains nodes such as\s+(?:\s*-\s+.+\s*\(.+\)\s*)+', re.IGNORECASE)
    script_pattern = re.compile(r'\s*- Script\d+ contains nodes such as\s+(?:\s*-\s+.+\s*\(.+\)\s*)+', re.IGNORECASE)
    node_pattern = re.compile(r'- (.*?) \((\d+)\)')

    node_counts = defaultdict(int)

    # Find all sections starting with '-Script'
    script_sections = script_pattern.findall(text)
    
    for section in script_sections:
        # Find all nodes within each section
        for match in node_pattern.finditer(section):
            node_type, count = match.groups()
            # Convert node type to lowercase to make the check case insensitive
            node_type_lower = node_type.strip().lower()
            node_counts[node_type_lower] += int(count)

    return node_counts

def consolidate_csv(filepath,csv_path):
    dict_count = {}
    with open(filepath,"r") as fi:
        
        lines = fi.readlines()
        for each_line in lines:
            if each_line not in dict_count:
                dict_count[each_line] = 0
            dict_count[each_line] +=1
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Node Type', 'Count'])
        for node_type, count in dict_count.items():
            csv_writer.writerow([node_type, count])
    return dict_count

# Function to write the nodes and counts to a CSV file
def write_nodes_to_csv(node_counts, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Node Type', 'Count'])
        for node_type, count in node_counts.items():
            csv_writer.writerow([node_type, count])


def write_to_file(output_file,txt_path):
    with open(output_file, 'w') as file:
        match_records = extract_match_file(txt_path)
        for script_header, node_matches in match_records:
            file.write(f"{script_header}\n")
            for node in node_matches:
                file.write(f"{node[1]}\n")

def plot_data(main_file_name,output_file_name,plot_title):
    df = pd.read_csv(main_file_name)
    
    # Plot the data
    plt.figure(figsize=(10, 8))
    plt.bar(df['Node_Type'], df['Count'], color='skyblue')
    plt.xlabel('Node')
    plt.ylabel('Count')
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file_name)

def plot_data_group(files_path,output_path):
    file_names = []
    for i in os.listdir(files_path):
        if len(i) > 1 and os.path.isfile(f'{files_path}/{i}'):
            file_names.append(i)
        else:
            continue
    
    if len(file_names) > 0:
        for index,filename in enumerate(file_names):
            main_file_names = f"{files_path}/{filename}"
            
            plot_data(main_file_names,f"{output_path}/count_plot_nodes_{index}.pdf","Nodes Count Plot")

def sort_csv(file_csv):
    df = pd.read_csv(file_csv)
    # Sort the DataFrame by the 'Count' column in descending order
    df_sorted = df.sort_values(by='Count', ascending=False)

    # Save the sorted DataFrame back to a CSV file
    sorted_csv_path = 'sorted_nodes_count_version4.csv'
    df_sorted.to_csv(sorted_csv_path, index=False)

def split_csv(input_file, output_prefix, lines_per_file=20):
    print(input_file)
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        # Read all rows into a list
        rows = list(reader)
        # Calculate the number of output files needed
        num_files = (len(rows) + lines_per_file - 1) // lines_per_file
        for i in range(num_files):
            start_index = i * lines_per_file
            end_index = start_index + lines_per_file
            print(end_index)
            output_rows = rows[start_index:end_index]
            
            output_file = f"{output_prefix}_{i+1}.csv"

            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(output_rows)
                
            

def main(pdf_path, csv_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(text)
    node_counts = extract_nodes(cleaned_text)
    #write_nodes_to_csv(node_counts, csv_path)
    print(f"CSV file '{csv_path}' has been created successfully.")

# Example usage
'''
if __name__ == "__main__":
    pdf_path = '/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/UnderstandingScratchRevisonsTagging2.pdf'  # Replace with your PDF file path
    csv_path = 'nodes_count.csv'
    main(pdf_path, csv_path)
'''
#extract_match_file('/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/revision_changes_description.txt',"all_matched_nodes_main_version6.csv")
plot_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main.csv","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/output_nodes_plot/least20_nodes_plot.pdf","Least 20 Nodes Count Plot")
#sort_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main_version6.csv")
#consolidate_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes.txt","counted_nodes.csv")
#split_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main_version6.csv","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/inputs_nodes_folder/all_nodes_split_versions")
#plot_data_group("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/inputs_nodes_folder","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/output_nodes_plot")