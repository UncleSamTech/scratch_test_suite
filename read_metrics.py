import re

def extract_metrics(file_path):
    # Initialize empty lists for each metric
    accuracy_nltk = []
    precision_nltk = []
    recall_nltk = []
    f1_score_nltk = []
    evaluation_time_nltk = []
    training_time_nltk = []

    # Define regex patterns to match each metric
    accuracy_pattern = re.compile(r"Accuracy:\s([0-9.]+)")
    precision_pattern = re.compile(r"Precision:\s([0-9.]+)")
    recall_pattern = re.compile(r"Recall:\s([0-9.]+)")
    f1_score_pattern = re.compile(r"F1 Score:\s([0-9.]+)")
    training_time_pattern = re.compile(r"Training time:\s([0-9.]+)\sseconds")
    evaluation_time_pattern = re.compile(r"Evaluation time:\s([0-9.]+)\sseconds")

    # Open the file and process line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Extract each metric using regex and add to respective lists
            accuracy_match = accuracy_pattern.search(line)
            precision_match = precision_pattern.search(line)
            recall_match = recall_pattern.search(line)
            f1_score_match = f1_score_pattern.search(line)
            evaluation_time_match = evaluation_time_pattern.search(line)
            training_time_match = training_time_pattern.search(line)

            if accuracy_match:
                accuracy_nltk.append(float(accuracy_match.group(1)))
            if precision_match:
                precision_nltk.append(float(precision_match.group(1)))
            if recall_match:
                recall_nltk.append(float(recall_match.group(1)))
            if f1_score_match:
                f1_score_nltk.append(float(f1_score_match.group(1)))
            if evaluation_time_match:
                evaluation_time_nltk.append(float(evaluation_time_match.group(1)))
            if training_time_match:
                training_time_nltk.append(float(training_time_match.group(1)))

    # Print the extracted lists or return them if needed
    print(f"data_10_projects_accuracy_nltk =  {accuracy_nltk}\n")
    print(f"data_10_projects_precision_nltk = {precision_nltk}\n")
    print(f"data_10_projects_recall_nltk = {recall_nltk}\n")
    print(f"data_10_projects_f1_score_nltk = {f1_score_nltk}\n")
    print(f"data_10_projects_evaluation_time_nltk = {evaluation_time_nltk}\n")
    print(f"data_10_projects_training_time_nltk = {training_time_nltk}")

    return accuracy_nltk, precision_nltk, recall_nltk, f1_score_nltk, evaluation_time_nltk,training_time_nltk

# Usage
file_path = "/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/trained_data_prec_rec_acc_10_projects.txt"
file_path_50 = "/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/trained_data_prec_rec_acc_50_projects.txt"
extract_metrics(file_path)
extract_metrics(file_path_50)