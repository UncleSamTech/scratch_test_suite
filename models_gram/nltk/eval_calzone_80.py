import os
import re
import time
import kenlm
import psutil
from multiprocessing import Process, cpu_count


def predict_next_token_kenlm_upd(model, context,vocab_name):
        
        next_token_probabilities = {}
        
        
        with open(vocab_name, "r", encoding="utf8") as vocab_f:
                vocabulary = vocab_f.readlines()
                for candidate_word in vocabulary:
                    candidate_word = candidate_word.strip()
                    context_with_candidate = context + " " + candidate_word
                    next_token_probabilities[candidate_word] = model.score(context_with_candidate)
        
        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)
        top_10_tokens_scores = sorted(next_token_probabilities.items(), key=lambda item: item[1], reverse=True)[:10]
        #returns predicted next token and list of top 10 tokens and scores
        return predicted_next_token,top_10_tokens_scores


def check_available_rank(list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank
                    

def evaluate_model(vocab_path, model_path, test_data, new_log_path, model_number, ngram_order, run_number):
    print(f"Evaluating model {model_path} with vocab {vocab_path}")

    # Load the language model
    model_rec = kenlm.Model(model_path)
    
    start_time = time.time()
    with open(test_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            sentence_tokens = line.split()
            if len(sentence_tokens) < 2:
                continue
            
            for idx in range(1, len(sentence_tokens)):
                context = ' '.join(sentence_tokens[:idx])
                true_next_word = sentence_tokens[idx]
                predicted_next_word, top_10_tokens = predict_next_token_kenlm_upd(model_rec, context, vocab_path)
                rank = check_available_rank(top_10_tokens, true_next_word)

                # Save results to log file
                investig_path = f"{new_log_path}/kenlm_investigate_{model_number}_{ngram_order}_{run_number}_logs.txt"
                if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                    with open(investig_path, "a") as ip:
                        ip.write("query,expected,answer,rank,correct\n")
                with open(investig_path, "a") as inv_path_file:
                    inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")
    
    diff = time.time() - start_time
    print(f"time taken is for {model_path} is {diff}")

def predict_next_token_kenlm_upd_opt(model, context, vocab_name):
        """
        Predicts the next token based on the given context using a KenLM model.
        Optimized for performance by reducing file reads, batching, and efficient top-10 selection.
        """
        # Read the vocabulary file once (if not already cached)
        if not hasattr('vocabulary'):
            with open(vocab_name, "r", encoding="utf8") as vocab_f:
                vocabulary = [line.strip() for line in vocab_f.readlines()]

        # Precompute the context with a trailing space
        context_with_space = context + " "

        # Score all candidate words in a single pass
        next_token_probabilities = {}
        for candidate_word in vocabulary:
            context_with_candidate = context_with_space + candidate_word
            next_token_probabilities[candidate_word] = model.score(context_with_candidate)

        # Find the predicted next token
        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)

        # Find the top-10 tokens without sorting the entire vocabulary
        top_10_tokens_scores = []
        for token, prob in next_token_probabilities.items():
            if len(top_10_tokens_scores) < 10:
                top_10_tokens_scores.append((token, prob))
            else:
                # Replace the smallest probability in the top-10
                min_prob_index = min(range(10), key=lambda i: top_10_tokens_scores[i][1])
                if prob > top_10_tokens_scores[min_prob_index][1]:
                    top_10_tokens_scores[min_prob_index] = (token, prob)

        # Sort the top-10 tokens by probability (descending)
        top_10_tokens_scores.sort(key=lambda x: x[1], reverse=True)

        return predicted_next_token, top_10_tokens_scores

def check_available_rank(list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank

def evaluate_test_file_from_resume_upd(vocab_path, model_path, test_data, new_log_path, model_number, ngram_order, run_number):
         
        """
        Evaluate the test file, resuming from the last evaluated point for the matching log file.
        """
        # Load the language model
        model_rec = kenlm.Model(model_path)

        # Extract the base name of the test file
        test_file_name = os.path.basename(test_data)

        # Construct the expected log file name
        log_file_name = f"kenlm_investigate_{model_number}_{ngram_order}_{run_number}_logs.txt"
        log_file_path = os.path.join(new_log_path, log_file_name)

        # Check if the log file exists
        if not os.path.exists(log_file_path):
            print(f"No matching log file found for test file {test_file_name}.")
            return

        print(f"Processing log file: {log_file_path}")

        # Count the number of log entries already generated
        log_entry_count = count_log_entries(log_file_path)

        # Find the resume point in the test file
        resume_point = find_resume_point(test_data, log_entry_count)
        if resume_point is None:
            print(f"Evaluation is already complete for log file {log_file_path}.")
            return

        line_num, token_pos = resume_point
        print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")

        start_time = time.time()

        # Open files for reading and appending
        with open(test_data, 'r', encoding="utf-8") as test_file:
            # Skip lines until the resume point, +1 to accommodate the header
            for _ in range(line_num + 1):
                next(test_file)

            # Process the remaining lines
            for line in test_file:
                tokens = line.strip().split()
                if len(tokens) >= 2:  # Only evaluate lines with 2 or more tokens
                    # Skip tokens until the resume point
                    for i in range(token_pos, len(tokens) - 1):
                        context = ' '.join(tokens[:i])
                        true_next_word = tokens[i]
                        predicted_next_word, top_10_tokens = predict_next_token_kenlm_upd_opt(model_rec, context, vocab_path)
                        rank = check_available_rank(top_10_tokens, true_next_word)

                        # Save results to log file
                        with open(log_file_path, "a") as inv_path_file:
                            inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

                    token_pos = 1  # Reset token position after the first line

        diff = time.time() - start_time
        print(f"Time taken for {model_path} and log file {log_file_path} is {diff} seconds.")

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



def evaluate_all_models_in_folder_in_order_with_runs(testdir, vocab_folder, model_folder, new_log_path):
    # Get vocab and model files
    vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab")])
    model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa")])
    
    # Match vocab and model files by model number, ngram order, and run number
    vocab_model_pairs = []
    vocab_pattern = re.compile(r"kenln_(\d+)_(\d+)_(\d+)\.vocab")
    model_pattern = re.compile(r"kenln_(\d+)_(\d+)_(\d+)\.arpa")

    for vocab in vocab_files:
        vocab_match = vocab_pattern.match(vocab)
        if not vocab_match:
            continue
        model_number, ngram_order, run_number = vocab_match.groups()

        for model in model_files:
            model_match = model_pattern.match(model)
            if model_match and model_match.groups() == (model_number, ngram_order, run_number):
                vocab_model_pairs.append((vocab, model, model_number, ngram_order, run_number))
                break

    # Check CPU utilization and find underutilized cores
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    underutilized_cores = [i for i, percent in enumerate(cpu_percent) if percent < 5]

    # Distribute workload across underutilized cores
    processes = []
    for i, (vocab_name, model_name, model_number, ngram_order, run_number) in enumerate(vocab_model_pairs):
        vocab_path = os.path.join(vocab_folder, vocab_name)
        model_path = os.path.join(model_folder, model_name)
        test_data = os.path.join(testdir, f"scratch_test_set_{model_number}_{ngram_order}_{run_number}_proc.txt")
        
        # Assign to underutilized cores in a round-robin fashion
        core = underutilized_cores[i % len(underutilized_cores)]
        p = Process(target=evaluate_test_file_from_resume_upd, args=(vocab_path, model_path, test_data, new_log_path, model_number, ngram_order, run_number))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


evaluate_all_models_in_folder_in_order_with_runs("/home/siwuchuk/thesis_project/kenlm/output_test","/home/siwuchuk/thesis_project/kenlm/vocab_files/80","/home/siwuchuk/thesis_project/kenlm/arpa_files/80","/home/siwuchuk/thesis_project/kenlm/logs/80")

# Example usage
# evaluate_all_models_in_folder_in_order_with_runs(testdir, vocab_folder, model_folder, new_log_path)