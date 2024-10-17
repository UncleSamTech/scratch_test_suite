
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
    sorted_csv_path = 'sorted_unique_open_codes.csv'
    df_sorted.to_csv(sorted_csv_path, index=False)




def sum_count_column(file_path):
    total_count = 0
    
    # Open and read the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        # Sum the values in the "Count" column
        for row in csv_reader:
            total_count += int(row['Count'])
    print("total count",total_count)
    #total count 5094
    return total_count



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
                
def count_open_codes(input_file):
    opencode_stores = {"introducing control flow structure": ["when I receive <node_value>","broadcast <node_value>"],"introducing event listeners":["when this sprite clicked"],               
    "introducing decision flow":["if <condition> then block","if <condition> then block else block"],
    "creating and utilizing custom blocks":["custom script call","define <nodes_value>"],
    "assigning values to variable":["set <node_variable> to <value>","set size to <node_value>","set size to <value> %","set size to <node_value> %","set size to <node> %","set <node_variable> to <node_value>","set volume to <value> %","set <node_value> effect to <value>"],"adding iteration to a program":["for each <node_value> in <node_value>","for each <node_variable> in <node_value>"],
    "creating clones of nodes":["create clone of <node_value>"],
    "concatenation of nodes":["join <node_value> <node_value>","join <value> <node_value>","join <node_value> <value>",
                              "join <node_variable> <node_value>","join <node_variable> <value>",
                              "join <node_variable> <node_variable>","join <value> <value>",
                              "join <node_values> <node_values>","join <node_variable><value>"], 
                              "introducing arithmetic operations":["<value> + <value>","<node_value> + <node_value>",
                              "<node_value> + <value>","<node_variable> + <node_value>","<node_variable> + <value>"
                              ,"<value> + <node_value>","<value> + <value>","<node_variable> * <value>",
                              "<node_variable> * <node_variable>",
                              "<node_value> * <node_value>","<node_value> * <value>"]}
            

def main(pdf_path, csv_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(text)
    node_counts = extract_nodes(cleaned_text)
    #write_nodes_to_csv(node_counts, csv_path)
    print(f"CSV file '{csv_path}' has been created successfully.")


def extract_codes(text_parsed):
    # Regular expression to match the lines of the codes under "Version"
    pattern = r'\d+\.\s*(.*)'
    matches = re.findall(pattern, text_parsed)
    return [match.strip() for match in matches]

def load_file_count_generate_new_file(file_path,new_file_name):
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')

    count_open_codes = {}
    for text in df['Codes']:
        if isinstance(text, str):
            codes = extract_codes(text)
            for code in codes:
            
                if code in count_open_codes:
                    count_open_codes[code] += 1
                else:
                    count_open_codes[code] = 1
    result_df = pd.DataFrame(list(count_open_codes.items()), columns=['Code', 'Count'])
    result_df.to_csv(new_file_name, index=False)



# Function to extract tags after 'Tags:'
def extract_tags(code):
    text = str(code).replace('\n', ' ').replace('\r', ' ')
    all_matches = re.findall(r'Tags:\s*(.*?)(?=\s*Version|\s*$)', text, re.IGNORECASE | re.DOTALL)
    tags = []
    for match in all_matches:
        # Split tags by commas and strip spaces
        tags.extend([tag.strip() for tag in match.split(',')])
    return tags

def extract_axial_codes(file_sheet):
    df = pd.read_excel(file_sheet)
    # Extract tags from the 'Codes' column
    df['TagsList'] = df['Codes'].apply(extract_tags)

    # Flatten the list of all tags
    all_tags = [tag for sublist in df['TagsList'] for tag in sublist]

    # Count distinct tags
    tag_counts = pd.Series(all_tags).value_counts().reset_index()
    tag_counts.columns = ['AxialCodes', 'Count']

    # Write the result to a CSV file
    tag_counts.to_csv('axial_tags_count_8.csv', index=False)

   



# Example usage
'''
if __name__ == "__main__":
    pdf_path = '/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/UnderstandingScratchRevisonsTagging2.pdf'  # Replace with your PDF file path
    csv_path = 'nodes_count.csv'
    main(pdf_path, csv_path)
'''
#extract_match_file('/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/revision_changes_description.txt',"all_matched_nodes_main_version6.csv")
#plot_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main.csv","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/output_nodes_plot/least20_nodes_plot.pdf","Least 20 Nodes Count Plot")
#sort_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main_version6.csv")
#consolidate_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes.txt","counted_nodes.csv")
#split_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_matched_nodes_main_version6.csv","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/inputs_nodes_folder/all_nodes_split_versions")
#plot_data_group("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/inputs_nodes_folder","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/splitted_files5/output_nodes_plot")
#sum_count_column("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/sorted_nodes_count_version4.csv")
#load_file_count_generate_new_file("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/Opencoding.xlsx","new_unique_open_codes.csv")
#sort_csv("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/new_unique_open_codes.csv")
extract_axial_codes("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/Opencoding_1.xlsx")