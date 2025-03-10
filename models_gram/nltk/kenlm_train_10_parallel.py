import os
import re
import time
import kenlm
import psutil
from multiprocessing import Pool, cpu_count

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_model(self, args):
        vocab_path, model_path, test_data, new_log_path, model_number, ngram_order, run_number = args

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
                    predicted_next_word, top_10_tokens = self.predict_next_token_kenlm_upd(model_rec, context, vocab_path)
                    rank = self.check_available_rank(top_10_tokens, true_next_word)

                    # Save results to log file
                    investig_path = f"{new_log_path}/kenlm_investigate_{model_number}_{ngram_order}_{run_number}_logs.txt"
                    if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                        with open(investig_path, "a") as ip:
                            ip.write("query,expected,answer,rank,correct\n")
                    with open(investig_path, "a") as inv_path_file:
                        inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

        diff = time.time() - start_time
        print(f"Time taken for {model_path} is {diff}")

    def evaluate_all_models_in_folder_in_order_with_runs(self, testdir, vocab_folder, model_folder, new_log_path):
        # Get vocab and model files
        om_voc = ['kenln_10_6_2.vocab', 'kenln_10_6_3.vocab', 'kenln_10_6_4.vocab', 'kenln_10_6_5.vocab']
        om_arp = ['kenln_10_6_2.arpa', 'kenln_10_6_3.arpa', 'kenln_10_6_4.arpa', 'kenln_10_6_5.arpa']
        vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab") and f in om_voc])
        model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa") and f in om_arp])

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

        # Prepare arguments for multiprocessing
        args_list = []
        for vocab_name, model_name, model_number, ngram_order, run_number in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            model_path = os.path.join(model_folder, model_name)
            test_data = os.path.join(testdir, f"scratch_test_set_{model_number}_{ngram_order}_{run_number}_proc.txt")
            args_list.append((vocab_path, model_path, test_data, new_log_path, model_number, ngram_order, run_number))

        # Distribute workload across CPU cores with less than 10% utilization
        pool = Pool(processes=self.get_available_cores())
        pool.map(self.evaluate_model, args_list)
        pool.close()
        pool.join()

    def get_available_cores(self):
        available_cores = []
        for core in range(cpu_count()):
            if psutil.cpu_percent(interval=0.1, percpu=True)[core] < 10:
                available_cores.append(core)
        return len(available_cores) if available_cores else 1

    def predict_next_token_kenlm_upd(self,model, context,vocab_name):
        
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

    def check_available_rank(self,list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank

# Example usage
evaluator = ModelEvaluator()
evaluator.evaluate_all_models_in_folder_in_order_with_runs("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test", "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/10", "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/10", "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/logs/10")