import re
import csv

# File paths
input_file = "/content/models_metrics_data/kenlm/kenlnm_acc_prec_rec_f1_portion.txt"  # Replace with the path to your input file
output_file = "/content/models_metrics_data/kenlm/KenLM_evaluation_metrics_10.csv"  # Replace with the desired output CSV file

# Read the input file
with open(input_file, "r") as file:
    lines = file.readlines()

# Prepare the CSV headers
headers = ["run", "ngram", "accuracy", "precision", "recall", "f1score", "evaluation_time"]

# Prepare the data
data = []
for line in lines:
    # Extract values using regex
     match = re.match(
        r"Run\s+(\d+)\s+for\s+\d+\s+projects\s+Vocab\s+name:\s+.*/kenln_order(\d+)\.vocab\s+\|\s+Accuracy:\s+([\d.]+)\s+\|\s+Precision:\s+([\d.]+)\s+\|\s+Recall:\s+([\d.]+)\s+\|\s+F1-score:\s+([\d.]+)\s+\|\s+Evaluation\s+time:\s+([\d.]+)\s+seconds",
        #r"Run\s+(\d+)\s+for\s+Vocab:\s+kenln_order(\d+)\.vocab\s+Model:\s+kenln_order\2\.arpa\s+\|\s+Accuracy:\s+([\d.]+)\s+\|\s+Precision:\s+([\d.]+)\s+\|\s+Recall:\s+([\d.]+)\s+\|\s+F1-score:\s+([\d.]+)\s+\|\s+Evaluation\s+time:\s+([\d.]+)\s+seconds",
        line,)
     if match:
        run, ngram, accuracy, precision, recall, f1score, eval_time = match.groups()
        data.append([run, ngram, accuracy, precision, recall, f1score, eval_time])

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data)    # Write the data

print(f"CSV file has been saved to {output_file}")