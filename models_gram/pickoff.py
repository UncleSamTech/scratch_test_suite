def count_log_entries(log_file_path):
    """Count the number of lines in the log file."""
    with open(log_file_path, 'r') as log_file:
        return sum(1 for line in log_file)

def count_expected_log_entries(test_file_path):
    """Count the total number of log entries that would be generated for the test file."""
    expected_entries = 0
    with open(test_file_path, 'r') as test_file:
        for line in test_file:
            tokens = line.strip().split()
            if len(tokens) >= 2:  # Only consider lines with 2 or more tokens
                expected_entries += len(tokens) - 1  # Tokens after the first token
    return expected_entries

def find_resume_point(test_file_path, log_entry_count):
    """Find the line and token position in the test file to resume evaluation."""
    with open(test_file_path, 'r') as test_file:
        current_log_entries = 0
        for line_num, line in enumerate(test_file):
            tokens = line.strip().split()
            if len(tokens) >= 2:  # Only consider lines with 2 or more tokens
                tokens_after_first = len(tokens) - 1
                if current_log_entries + tokens_after_first >= log_entry_count:
                    # Resume point is in this line
                    token_pos = log_entry_count - current_log_entries
                    return line_num, token_pos
                current_log_entries += tokens_after_first
    return None  # If no resume point is found

def evaluate_test_file(test_file_path, log_file_path, output_file_path):
    """Evaluate the test file, resuming from the last evaluated point."""
    # Count the number of log entries already generated
    log_entry_count = count_log_entries(log_file_path)
    
    # Find the resume point in the test file
    resume_point = find_resume_point(test_file_path, log_entry_count)
    if resume_point is None:
        print("Evaluation is already complete.")
        return
    
    line_num, token_pos = resume_point
    print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")

    # Open files for reading and appending
    with open(test_file_path, 'r') as test_file, open(log_file_path, 'a') as log_file, open(output_file_path, 'a') as output_file:
        # Skip lines until the resume point, +1 to accomodate the header
        for _ in range(line_num + 1):
            next(test_file)
        
        # Process the remaining lines
        for line in test_file:
            tokens = line.strip().split()
            if len(tokens) >= 2:  # Only evaluate lines with 2 or more tokens
                # Skip tokens until the resume point
                for i in range(token_pos, len(tokens) - 1):
                    context = ' '.join(tokens[:i + 1])
                    next_token = tokens[i + 1]
                    # Simulate evaluation (replace with your actual evaluation logic)
                    result = f"{context},{next_token},<s>,{i + 2},0"
                    log_file.write(result + '\n')
                    output_file.write(result + '\n')
                token_pos = 0  # Reset token position after the first line

# Example usage
test_file_path = 'test_file.txt'
log_file_path = 'log_file.txt'
output_file_path = 'output_file.txt'

evaluate_test_file(test_file_path, log_file_path, output_file_path)