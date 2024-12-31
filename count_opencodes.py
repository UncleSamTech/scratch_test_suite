import pandas as pd
import re

# Load the CSV file
data = pd.read_csv('/Users/samueliwuchukwu/downloads/Opencodings.csv')

# Function to extract and count codes from the 'Codes' column
def parse_and_count_codes(codes_column):
    code_counts = {}

    for entry in codes_column.dropna():  # Ignore NaN values
        versions = re.findall(r'Version \d+.*?(?=Version \d+|$)', entry, re.DOTALL)
        for version in versions:
            codes = re.findall(r'\d+\.\s+([^\n]+)', version)
            for code in codes:
                code = code.strip()
                code_counts[code] = code_counts.get(code, 0) + 1

    return code_counts

# Parse the 'Codes' column
code_counts = parse_and_count_codes(data['Codes'])

# Convert the counts to a DataFrame
result_df = pd.DataFrame(list(code_counts.items()), columns=['Code', 'Count'])

# Sort the DataFrame by 'Count' in descending order
result_df = result_df.sort_values(by='Count', ascending=False)

# Save the result to a new CSV file
output_path = '/Users/samueliwuchukwu/downloads/Distinct_Codes_Count_summary.csv'
result_df.to_csv(output_path, index=False)

# Generate summary statistics
summary_stats = result_df['Count'].describe()

print(f"Distinct codes and their counts have been saved to: {output_path}")
print("\nSummary Statistics:\n")
print(summary_stats)

# count    199.000000
# mean       2.859296
# std        4.274697
# min        1.000000
# 25%        1.000000
# 50%        1.000000
# 75%        3.000000
# max       30.000000
