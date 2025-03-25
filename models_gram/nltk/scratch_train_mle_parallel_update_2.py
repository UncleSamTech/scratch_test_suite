import os
import pickle
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
import nltk
import time
import matplotlib.pyplot as plt
#import pandas as pd
import random
import scipy.stats as stats
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_curve,f1_score,confusion_matrix
import heapq
from random import sample
import seaborn as sns
import numpy as np
import pandas as pd
import re
from itertools import product
import multiprocessing
import psutil
import time
import itertools
import gc

class scratch_train_mle:

    def __init__(self):
        self.scratch_model = None
        self.ngram_model = None
        self.loaded_scratch_model = None

    def train_mle(self,train_data,n,model_name):

        with open(train_data,"r",encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            
            tokenized_scratch_data = [list(word_tokenize(sent.strip())) for sent in lines]
            train_data_val,padded_sents = padded_everygram_pipeline(n,tokenized_scratch_data)
        
        try:
            self.scratch_model = MLE(n)
            self.scratch_model.fit(train_data_val,padded_sents)

            with open(f'{model_name}_{n}.pkl',"wb") as fd:
                pickle.dump(self.scratch_model,fd)
        except Exception as es:
            print("error as a result of ", es)



    def train_mle_new(self, train_data, n, model_name, model_path, model_number, run):
        try:
            with open(train_data, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            tokenized_scratch_data = self.gener_list_list(lines)

            

            train_data_val, padded_sents = padded_everygram_pipeline(n, tokenized_scratch_data)

            scratch_model = MLE(n)
            scratch_model.fit(train_data_val, padded_sents)

            # Ensure the directory exists
            os.makedirs(model_path, exist_ok=True)

            model_file = f"{model_path}/{model_name}{model_number}_{n}_{run}.pkl"
            if not os.path.exists(model_file):
                print(f"Saving model to {model_file}")  # Debugging

                with open(model_file, "wb") as fd:
                    pickle.dump(scratch_model, fd)
            else:
                print(f"model {model_file} already trained")

        except Exception as e:

            print("Error:", e)

    def gener_list_list(self,data):
        return [list(word_tokenize(line.strip())) for line in data if line.strip()]



    def extract_vocabulary_nltk(self,model_name):
        # Load the trained MLE model from the pickle file
        model = self.load_trained_model(model_name)
    
        # Access the vocabulary (as an iterable) and convert it to a list
        vocab_list = list(model.vocab)
    
        return vocab_list
    

           
    def load_trained_model(self,model_name) :
        with open(model_name,"rb") as f:
            self.loaded_scratch_model = pickle.load(f)
            #print(type(self.loaded_scratch_model))
            #print(self.loaded_scratch_model.vocab)
            #print(self.loaded_scratch_model.counts("event_whenflagclicked"))
            #print(self.loaded_scratch_model.score("event_whenflagclicked"))
            #print(self.loaded_scratch_model.vocab.lookup("move"))
        return self.loaded_scratch_model
    

    def compute_confusion_matrix(self, y_true, y_pred, result_path, proj_number,ngram,run,top_k=10):
        labels = np.unique(np.concatenate((y_true, y_pred)))  # Get unique labels
        id2label = {i: str(label) for i, label in enumerate(labels)}  # Map indices to labels
        label2id = {v: k for k, v in id2label.items()}  # Reverse mapping (if needed)

        # Compute confusion matrix
        print("\nComputing Confusion Matrix...")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        num_classes = conf_matrix.shape[0]
        print(f" number of classes {num_classes}")
        metrics = {id2label[i]:{"TP":0,"FP":0,"FN":0,"TN":0} for i in range(num_classes)}
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for i in range(num_classes):
            TP = conf_matrix[i,i]
            FP = np.sum(conf_matrix[:,i]) - TP
            FN = np.sum(conf_matrix[i, :]) - TP
            TN = np.sum(conf_matrix) - (TP + FP + FN)

            label = id2label[i]
            metrics[label]["TP"] = TP
            metrics[label]["FP"] = FP
            metrics[label]["FN"] = FN
            metrics[label]["TN"] = TN

            total_tp += TP
            total_fp += FP
            total_fn += FN
            total_tn += TN

        # Write metrics to file and print
        with open(f"{result_path}/tp_fp_fn_tn_label_val_{ngram}_{run}.csv", "w") as af:
            af.write("Class,TP,FP,FN,TN\n")  # Header
            for label, values in metrics.items():
                #print(f"Label {label}: TP={values['TP']}, FP={values['FP']}, FN={values['FN']}, TN={values['TN']}")
                af.write(f"{label},{values['TP']},{values['FP']},{values['FN']},{values['TN']}\n")

        # Print total metrics
        #print(f"\nTotal TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
        #print(f"Confusion Matrix:\n{conf_matrix}")
        with open(f"{result_path}/total_results_nltk_tp_tn_fp_fn_{ngram}_{run}.csv","w") as tot:
          tot.write("total_tn,total_fp,total_fn,total_tp,no_of_classes\n")
          tot.write(f"{total_tn},{total_fp},{total_fn},{total_tp},{num_classes}")

        conf_matrix = np.array([[total_tp, total_fn],
                            [total_fp, total_tn]])

        # Plotting the confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()

        # # Get the unique class labels in sorted order (this will be used for indexing)
        # unique_classes = np.unique(np.concatenate((y_true, y_pred)))  # Combine y_true and y_pred to cover all classes

        # # Determine the top-k most frequent classes based on y_true
        # class_counts = pd.Series(y_true).value_counts().head(top_k).index

        # # Map the class labels to indices based on the sorted unique classes
        # class_indices = [np.where(unique_classes == label)[0][0] for label in class_counts]

        # # Use np.ix_ to index into the confusion matrix
        # filtered_conf_matrix = conf_matrix[np.ix_(class_indices, class_indices)]

        # # Optional: Save confusion matrix as a heatmap
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(filtered_conf_matrix, annot=True, fmt='d', cmap='Blues',
        #         xticklabels=class_counts, yticklabels=class_counts)

        # # Rotate x-axis labels to avoid overlap
        # plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
        # plt.yticks(rotation=0)  # Keep y-axis labels as they are

        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.title(f'Confusion Matrix (Top {top_k} Classes)')
        # # Adjust layout to make sure everything fits
        # plt.tight_layout()
        plt.savefig(f"{result_path}/confusion_matrix_run_an_nltk_tp_tn_fp_fn{proj_number}_{ngram}_{run}.pdf")
        plt.close()


    def predict_next_scratch_token(self,model_name,context_data):
        loaded_model = self.load_trained_model(model_name)
        scratch_next_probaility_tokens = {}
        

        for prospect_token in loaded_model.vocab:
            
            scratch_next_probaility_tokens[prospect_token] = loaded_model.score(prospect_token,context_data.split(" "))
        
        scratch_predicted_next_token = max(scratch_next_probaility_tokens,key=scratch_next_probaility_tokens.get)
        
        
        #print("predicted score ", scratch_next_probaility_tokens)
        #scratch_predicted_next_token = scratch_predicted_next_token
        return scratch_predicted_next_token
    
    def check_vocab_size(self,model_name):
        ld_md = self.load_trained_model(model_name)
        vocab_size = len(ld_md.vocab)
        for each_token in ld_md.vocab:
            print(each_token)
        print(vocab_size) 


    def predict_next_scratch_token_upd(self,model_name,context_data):
        loaded_model = self.load_trained_model(model_name)
        scratch_next_probaility_tokens = {}
        

        for prospect_token in loaded_model.vocab:
            
            scratch_next_probaility_tokens[prospect_token] = loaded_model.score(prospect_token,context_data.split(" "))
        
        scratch_predicted_next_token = max(scratch_next_probaility_tokens,key=scratch_next_probaility_tokens.get)
        top_10_tokens_scores = sorted(scratch_next_probaility_tokens.items(), key=lambda item: item[1], reverse=True)[:10]
        
        #print("predicted score ", scratch_next_probaility_tokens)
        #scratch_predicted_next_token = scratch_predicted_next_token
        return scratch_predicted_next_token,top_10_tokens_scores
    
    def predict_next_scratch_token_upd_opt(self, model_name, context_data):
        loaded_model = self.load_trained_model(model_name)
        context_tokens = context_data.split()  # Avoid repeated splits

        scratch_next_probaility_tokens = {
            token: loaded_model.score(token, context_tokens)
            for token in loaded_model.vocab
        }

        # Get the top predicted token
        scratch_predicted_next_token = max(scratch_next_probaility_tokens, key=scratch_next_probaility_tokens.get)

        # Get the top 10 tokens (sorted only once)
        top_10_tokens_scores = sorted(scratch_next_probaility_tokens.items(), key=lambda x: x[1], reverse=True)[:10]

        return scratch_predicted_next_token, top_10_tokens_scores

    
    def check_available_rank(self,list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank
    
    def check_available_rank_opt(self, list_tuples, true_word):
        true_word = true_word.strip()  # Strip once outside the loop
        
        for ind, (word, _) in enumerate(list_tuples, start=1):  # Unpack tuple & start index at 1
            if word.strip() == true_word:
                return ind  # Return immediately on match
                
        return -1  # Return -1 if not found

    
    def compute_score(self,model_name,token,context_data):
        load_model = self.load_trained_model(model_name)
        token = token.strip()
        return load_model.score(token, context_data.split(" "))
    

    def evaluate_mrr_nltk(self,model_name,result_path,split_folder,proj_number):
        # Ensure result path exists
        eval_files = ["scratch_test_data_chunk_1.txt","scratch_test_data_chunk_10.txt","scratch_test_data_chunk_11.txt"]
        os.makedirs(result_path, exist_ok=True)
        all_vocab = self.extract_vocabulary_nltk(model_name) 
        
        # Process each file in the split folder
        for split_file in sorted(os.listdir(split_folder)):
            split_file_path = os.path.join(split_folder, split_file)
            if not os.path.isfile(split_file_path):
                continue
            
            split_file = split_file.strip()
            # if split_file in eval_files:
            #     continue

            total_cumulative_rr = 0
            total_count = 0

            start_time = time.time()


            # Process each line in the file
            with open(split_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    # Preprocess the line
                    #line = line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower()
                    sentence_tokens = line.split(" ")
                    if len(sentence_tokens) < 2:
                        continue
                    
                    for idx in range(1,len(sentence_tokens)):
                        context = " ".join(sentence_tokens[:idx])
                        true_next_word = sentence_tokens[idx]

                        # Compute scores for tokens
                        heap = []
                        for token in all_vocab:
                            context_score = self.compute_score(model_name, token, context)
                            if len(heap) < 10:
                                heapq.heappush(heap, (context_score, token))
                            elif context_score > heap[0][0]:
                                heapq.heappushpop(heap, (context_score, token))

                        heap.sort(reverse=True, key=lambda x: x[0])
                        token_ranks = {t: rank + 1 for rank, (score, t) in enumerate(heap)}

                        # Compute reciprocal rank
                        true_next_word = true_next_word.strip()
                        rank = token_ranks.get(true_next_word, 0)
                        if rank:
                            current_rank = 1 / rank
                            total_cumulative_rr += current_rank
                            #print(f"processed line {line} with reciprocal rank {current_rank} and total cummulative {total_cumulative_rr}")
                        total_count += 1
                    
            
            # Calculate total RR and lines for the file
            time_spent = time.time() - start_time
            result_file = os.path.join(result_path, f"nltk_rr_results_{proj_number}_order.txt")
            with open(result_file, "a") as rf:
                rf.write(f"File name : {split_file}\n")
                rf.write(f"Total Reciprocal Rank: {total_cumulative_rr}\n")
                rf.write(f"Total Lines: {total_count}\n")
                rf.write(f"Time Spent: {time_spent:.2f} seconds\n")
            

            print(f"Processed {split_file}: RR = {total_cumulative_rr}, Lines = {total_count}")



    
    def scratch_evaluate_model_nltk(self,test_data,model_name,result_path,proj_number,ngram,run):

        y_true = []
        y_pred = []

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()

                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1].lower()

                predicted_next_word = self.predict_next_scratch_token(model_name,context)
                
                y_true.append(true_next_word)
                y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)
        self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,ngram,run)
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
    

    def scratch_evaluate_model_nltk_in_order_all(self,test_data,model_path,proj_number,result_path):
        all_models = sorted([f for f in os.listdir(model_path) if f.endswith(".pkl")])

        y_true = []
        y_pred = []
        log_file = f"{result_path}logs/trained_data_prec_rec_acc_{proj_number}_projects.txt"
        #log_file_error = f"{result_path}logs/trained_data_prec_rec_acc_{proj_number}_projects_error.txt"


        for each_model in all_models:

            match = re.search(r"_(\d+)\.pkl$", each_model.strip())
            ngram = match.group(1)
            for each_run in range(1,6):
                eval_start_time = time.time()  
                with open(test_data,"r",encoding="utf-8") as f:
                    lines= f.readlines()
                    random.shuffle(lines)
                    lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            
                    for line in lines:
                        line = line.strip()
                        sentence_tokens = line.split()

                        if len(sentence_tokens) < 2:
                            continue
                        
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])  # Use all words except the last one as context
                            true_next_word = sentence_tokens[idx]
                            each_model = each_model.strip()
                            predicted_next_word = self.predict_next_scratch_token(each_model,context)
                
                            y_true.append(true_next_word)
                            y_pred.append(predicted_next_word)
                eval_duration = time.time() - eval_start_time

                #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
                f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)
                if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
                        with open(log_file,"a") as fl:
                            fl.write(f"ngram,run,accuracy,precision,recall,f1score,training_time,evaluation_time\n")

                with open(log_file, "a") as precs:
                    precs.write(f"{ngram},{each_run},{accuracy},{precision},{recall},{f1score},{eval_duration}\n")
        

                self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,ngram,each_run)


    def scratch_evaluate_model_nltk_in_order_all_upd_norun(self,test_data,model_path,proj_number,result_path):
        all_models = sorted([f for f in os.listdir(model_path) if f.endswith(".pkl")])

        

        for each_model in all_models:

            match = re.search(r"_(\d+)\.pkl$", each_model.strip())
            ngram = match.group(1)
            eval_start_time = time.time()  
            with open(test_data,"r",encoding="utf-8") as f:
                    lines= f.readlines()
                    random.shuffle(lines)
                    
                    for line in lines:
                        line = line.strip()
                        sentence_tokens = line.split()

                        if len(sentence_tokens) < 2:
                            continue
                        
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])  # Use all preceding tokens as context
                            true_next_word = sentence_tokens[idx]
                            each_model_path = f"{model_path}/{each_model.strip()}"
                            predicted_next_word,top_10_tokens = self.predict_next_scratch_token_upd(each_model_path,context)
                            rank = self.check_available_rank(top_10_tokens,true_next_word.strip())
                            eval_duration = time.time() - eval_start_time
                            log_file = f"{result_path}/nltk_investigate_{proj_number}_{ngram}.txt"
                            if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
                                with open(log_file,"a") as fl:
                                    fl.write(f"query,expected,answer,rank,correct\n")
                            
                            with open(log_file, "a") as precs:
                                precs.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

            eval_duration = time.time() - eval_start_time
            print(f"total duration for evaluating ngram {ngram} is {eval_duration:.2f}")           

                
    def count_log_entries(self,log_file_path):
        """Count the number of lines in the log file."""
        with open(log_file_path, 'r') as log_file:
            total = sum(1 for line in log_file)
            #to exclude the header line
            print(f"total logs so far is {total}")
            return  total - 1
        
    def count_expected_log_entries(self,test_file_path):
        """Count the total number of log entries that would be generated for the test file."""
        expected_entries = 0
        with open(test_file_path, 'r') as test_file:
            for line in test_file:
                tokens = line.strip().split()
                if len(tokens) >= 2:  # Only consider lines with 2 or more tokens
                    expected_entries += len(tokens) - 1  # Tokens after the first token
        return expected_entries
    
    def find_resume_point(self,test_file_path, log_entry_count):
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
            print(f"total lines in test file is {current_log_entries} ")
        return None  # If no resume point is found


 
    def scratch_evaluate_model_nltk_in_order_all_new(self, test_data, model_name, result_path, model_path, run, n, model_number):
        log_file = f"{result_path}/nltk_investigate_{model_number}_{n}_{run}_logs.txt"
        formed_model = f"{model_path}/{model_name}{model_number}_{n}_{run}.pkl"

        # Check if the log file needs a header
        file_needs_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0

        # Open the log file in append mode
        with open(log_file, "a", encoding="utf-8") as log_f:
            if file_needs_header:
                log_f.write("query,expected,answer,rank,correct\n")

            # Open the test data file
            with open(test_data, "r", encoding="utf-8") as test_f:
                # Determine the resume point if the file doesn't need a header
                if not file_needs_header:
                    log_entry_count = self.count_log_entries(log_file)
                    resume_point = self.find_resume_point(test_data, log_entry_count)
                    if resume_point is None:
                        print("Evaluation completed")
                        return

                    line_num, token_pos = resume_point
                    print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")
                    skipped_lines = itertools.islice(test_f, line_num, None)
                else:
                    skipped_lines = test_f
                    token_pos = 1  # Start from the first token if no resume point

                # Process each line in the test data
                for line in skipped_lines:
                    line = line.strip()
                    sentence_tokens = line.split()
                    if len(sentence_tokens) < 2:
                        continue  # Skip empty or single-word lines

                    # Process each token in the sentence
                    for i in range(token_pos, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:i])
                        true_next_word = sentence_tokens[i]

                        # Predict the next word and get the rank
                        predicted_next_word, top_10_tokens = self.predict_next_scratch_token_upd_opt(formed_model, context)
                        rank = self.check_available_rank_opt(top_10_tokens, true_next_word)

                        # Write the result to the log file
                        log_f.write(f"{context},{true_next_word},{predicted_next_word},{rank},{1 if true_next_word == predicted_next_word else 0}\n")

                    token_pos = 1  # Reset token position after processing the first resumed line

        # Clean up
        del formed_model
        gc.collect()



    def scratch_evaluate_model_nltk_in_order_all_new(self, test_data, model_name, result_path,model_path,run,n,model_number):


        log_file = f"{result_path}/nltk_investigate_{model_number}_{n}_{run}_logs.txt"
        formed_model = f"{model_path}/{model_name}{model_number}_{n}_{run}.pkl"
        file_needs_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        if file_needs_header:
            precs.write("query,expected,answer,rank,correct\n")

            with open(test_data, "r", encoding="utf-8") as f, open(log_file, "a") as precs:
                

                for line in f:
                    sentence_tokens = line.strip().split()
                    if len(sentence_tokens) < 2:
                        continue  # Skip empty or single-word lines

                    for idx in range(1, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:idx])
                        true_next_word = sentence_tokens[idx]

                        

                        predicted_next_word, top_10_tokens = self.predict_next_scratch_token_upd_opt(formed_model, context)
                        rank = self.check_available_rank_opt(top_10_tokens, true_next_word)

                        precs.write(f"{context},{true_next_word},{predicted_next_word},{rank},{1 if true_next_word == predicted_next_word else 0}\n")

        else:
            # Count the number of existing log entries
            log_entry_count = self.count_log_entries(log_file)

            # Find resume point
            resume_point = self.find_resume_point(test_data, log_entry_count)
            if resume_point is None:
                print("Evaluation completed")
                return

            line_num, token_pos = resume_point
            print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")

            with open(test_data, 'r') as test_file, open(log_file, "a") as inv_path_file:
                # Skip lines until the resume point
                skipped_lines = itertools.islice(test_file, line_num, None)

                for line in skipped_lines:
                    line = line.strip()
                    sentence_tokens = line.split()
                    if len(sentence_tokens) < 2:
                        continue

                    
                     
                    for i in range(token_pos, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:i])
                        true_next_word = sentence_tokens[i]

                        predicted_next_word, top_10_tokens = self.predict_next_scratch_token_upd_opt(formed_model, context)
                        rank = self.check_available_rank_opt(top_10_tokens, true_next_word)

                        inv_path_file.write(f"{context},{true_next_word},{predicted_next_word},{rank},{1 if true_next_word == predicted_next_word else 0}\n")

                        

                    token_pos = 1  # Reset token position after processing first resumed line
        
        del formed_model
        gc.collect()
           
                  
    def scratch_evaluate_model_nltk_in_order_all_new_opt(self, test_data, model_name, result_path, model_path, run, n, model_number):
        log_file = f"{result_path}/nltk_investigate_{model_number}_{n}_{run}_logs.txt"
        formed_model = f"{model_path}/{model_name}{model_number}_{n}_{run}.pkl"
        
        # Determine if we need to write header
        file_needs_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        
        # Get resume point if needed
        resume_point = None
        if not file_needs_header:
            log_entry_count = self.count_log_entries(log_file)
            resume_point = self.find_resume_point(test_data, log_entry_count)
            if resume_point is None:
                print("Evaluation completed")
                return

        # Open both files just once
        with open(test_data, "r", encoding="utf-8") as test_file, \
            open(log_file, "a", encoding="utf-8") as output_file:
            
            # Write header if needed
            if file_needs_header:
                output_file.write("query,expected,answer,rank,correct\n")
            
            # Initialize processing state
            current_line = 0
            processing_from_start = file_needs_header or resume_point is None

            # Skip to resume line if needed
            if resume_point:
                current_line, current_token_pos = resume_point
                print(f"Resuming evaluation from line {current_line + 1}, token position {current_token_pos + 1}.")
                # Skip to the resume line
                for _ in range(current_line):
                    next(test_file)
            
            # Process all lines from current position
            for line in test_file:
                line = line.strip()
                sentence_tokens = line.split()
                if len(sentence_tokens) < 2:
                    current_line += 1
                    continue

                # Determine start position for tokens:
                # - When starting fresh, skip first token (position 1)
                # - When resuming, use the saved token position
                token_start = 1 if processing_from_start and current_line == 0 else (
                    resume_point[1] if resume_point and current_line == resume_point[0] else 1
                )

                for idx in range(token_start, len(sentence_tokens)):
                    context = ' '.join(sentence_tokens[:idx])
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word, top_10_tokens = self.predict_next_scratch_token_upd_opt(formed_model, context)
                    rank = self.check_available_rank_opt(top_10_tokens, true_next_word)

                    output_file.write(f"{context},{true_next_word},{predicted_next_word},{rank},{1 if true_next_word == predicted_next_word else 0}\n")
                
                current_line += 1
                # After first line processed, always start from token position 1 for subsequent lines
                processing_from_start = False

        # Clean up
        del formed_model
        gc.collect()

    def scratch_evaluate_model_nltk_in_order(self,test_data,model_name,result_path,proj_number,ngram,run):

        y_true = []
        y_pred = []

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()

                if len(sentence_tokens) < 2:
                    continue

                #evaluate all tokens in order
                for idx in range(1,len(sentence_tokens)):
                    context = ' '.join(sentence_tokens[:idx])
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word = self.predict_next_scratch_token(model_name,context)
                
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)
        self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,ngram,run)
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
    

    def scratch_evaluate_model_nltk_in_order_upd(self,test_data,model_name,result_path,proj_number,ngram,run):

        y_true = []
        y_pred = []

        all_models = [model  for model in os.listdir(result_path) if model.endswith(".pkl")]
        if all_models:
            for each_model in all_models:

                with open(test_data,"r",encoding="utf-8") as f:
                    lines= f.readlines()
                    random.shuffle(lines)
                    #lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
                    
                    for line in lines:
                        line = line.strip()
                        sentence_tokens = line.split()

                        if len(sentence_tokens) < 2:
                            continue

                        #evaluate all tokens in order
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])
                            true_next_word = sentence_tokens[idx]
                            proper_model = f"{result_path}/{each_model}"

                            predicted_next_word,top10_tokens = self.predict_next_scratch_token(model_name,context)
                            rank = self.check_available_rank(top10_tokens)
                        
                            y_true.append(true_next_word)
                            y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)
        self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,ngram,run)
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
 
    
    def scratch_evaluate_model_nltk_first(self,test_data,model_name):

        y_true = []
        i=0
        y_pred = []
        context = None
        true_next_word = None
        predicted_next_word = None

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            #lines = [line.replace("_","UNDERSCORE") for line in lines]
            #lines = [line.replace("<","LEFTANG") for line in lines]
            #lines = [line.replace(">","RIGHTANG") for line in lines]
            random.shuffle(lines)
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()
                
                if len(sentence_tokens) > 1:
                    print("first word ", sentence_tokens[1])
                    context = ' '.join(sentence_tokens[0:-1])  # Use all words except the first one as context
                    true_next_word = sentence_tokens[1]
                    #print("true next word ", true_next_word)
            
                    predicted_next_word = self.predict_next_scratch_token(model_name,context)
                    #print(f"compare {true_next_word} with predicted next word {predicted_next_word}")
                    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/seelogs.txt","a") as fp:
                        fp.write(f"for context {context} next token {predicted_next_word}")
                        fp.write("\n")
                
                    print(f"predicted {predicted_next_word} true word {true_next_word}")
                
                    i+=1
                    if i%500 == 0:
                        print("see it",i)
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)
                else:
                    context = ' '.join(sentence_tokens)  # Use all words except the first one as context
                    true_next_word = sentence_tokens[0]
                    predicted_next_word = self.predict_next_scratch_token(model_name,context)

                    i+=1
                    if i%500 == 0:
                        print("see it",i)
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true,y_pred)
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
    
    def shuffle_test_data(self,input_string):
        if isinstance(input_string,str) and len(input_string) > 0:
            #convert to list
            list_string  = list(input_string)
            shuffled_list = random.shuffle(list_string)
            shuffled_res = ''.join(shuffled_list)
            return shuffled_res
    
    def convert_hours(self,seconds):
        return round((seconds / 3600),2)

    def plot_precision_recall_curve(self,plot_name):

        Accuracy = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367]
        Precision = [0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743,0.2264853658226743,0.22696845761011392]
        Recall = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367]
        F1 = [0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387,0.21718121691450387,0.2176643087019435]
        Ngrams = [2,3,4,5,6,7,8,9,10]

        Accuracy2 = [0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328,0.24589371980676328]
        Precision2 = [0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372,0.22721000350383372]
        Recall2 = [0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328,0.24589371980676328]
        F1_2 = [0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699,0.2179863698935699]
        Ngrams2 = [10,11,12,13,14,15]

        Accuracy3 = [0.025120772946859903,0.21690821256038648,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        Precision3  = [0.0033068915888476084,0.1932486508468289,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424]
        Recall3 = [0.025120772946859903,0.21690821256038648,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        F1_3 = [0.005844424726412303,0.1908012224104309,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736]
        Ngrams3 = [2,3,4,5,6,7,8,9]

        Accuracy4 = [0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        Precision4  = [0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424]
        Recall4 = [0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        F1_4 = [0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736]
        Ngrams4 = [10,11,12,13,14,15]
        

        Accuracy_plot2_8 = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407]
        Recall_plot2_8 = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407]
        Accuracy_plot9_15 = [0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328]
        Recall_plot9_15 = [0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328]
        Precision_plot2_8 = [0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743]
        Precision_plot9_15 = [0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372]
        f1_plot2_8 = [0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387]
        f1_plot9_15 = [0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699]
        

        ngram2_8 = list(range(2,9))
        ngram_9_15 = list(range(9,16))
        ngram_2_10 = list(range(2,11))
        ngram_11_19 = list(range(11,20))

        accuracy_2_10 = [0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        precision_2_10 = [0.00039637450273575164,0.001006502629642402,0.01410223779838277,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145]
        recall_2_10 = [0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        f1_2_10 = [0.0006595641494970354,0.0012696922764036857,0.01547662029383393,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823]

        accurracy_11_19 = [0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        precision_11_19 = [0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145]
        recall_11_19 = [0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        f1_11_19 = [0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823]
        
        main_accuracy = [0.009372597103015442,0.026268210062553254,0.02728651883871236,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844]
        main_precision = [0.00250684777717733,0.02000157711620491,0.02054685262657442,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945]
        main_recall = [0.009372597103015442,0.026268210062553254,0.02728651883871236,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844]
        main_f1 = [0.0033898931640232023,0.019506147406811364,0.020210073337690322,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567]
        main_training_time = [self.convert_hours(177.9156),self.convert_hours(249.6270),self.convert_hours(349.6335),self.convert_hours(475.4877),self.convert_hours(642.8615),self.convert_hours(861.8450),self.convert_hours(1094.8200),self.convert_hours(1372.9215),self.convert_hours(1628.4691)]
        main_evaluation_time = [self.convert_hours(512.2936),self.convert_hours(1200.4897),self.convert_hours(2738.8720),self.convert_hours(5382.5739),self.convert_hours(9082.1960),self.convert_hours(13952.8228),self.convert_hours(20310.5438),self.convert_hours(24475.8971),self.convert_hours(34695.9437)]
        main_gram_range = list(range(2,11))

        plt.plot(main_gram_range, main_precision, label = "Precision")
        plt.plot(main_gram_range, main_recall, label = "Recall")
        plt.plot(main_gram_range,main_f1, label = "F1")
        plt.plot(main_gram_range, main_accuracy, label = "Accuracy")
        #plt.plot(main_gram_range,main_training_time,label="Training Time converted to hrs")
        #plt.plot(main_gram_range,main_evaluation_time,label="Evaluation Time converted to hrs")
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Nltk_Model Scores vs N-Gram Orders 2 - 10 on the portion')
        plt.legend()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}_2_10.pdf')
        #plt.show()

    def paired_t_test(self,nltk_2_10,nltk_11_19):
        if isinstance(nltk_2_10,list) and len(nltk_2_10) > 0 and isinstance(nltk_11_19,list) and len(nltk_11_19) > 0:
            test_val = stats.ttest_rel(nltk_2_10,nltk_11_19)
            #print(test_val)
            return test_val
        
    def wilcon_t_test(self,group1,group2):
        return stats.wilcoxon(group1,group2)
        
    def multiple_train(self,list_ngrams,test_data,model_name,train_data):
        final_result = {}
        for each_gram in list_ngrams:
            try:
                self.train_mle(train_data,each_gram,model_name)
                acc,precision,rec,f1_score = self.scratch_evaluate_model_nltk(test_data,f'{model_name}_{each_gram}.pkl')

                final_result[f'{each_gram}-gram_nltk'] = [acc,precision,rec,f1_score]
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/trained_data_prec_rec_acc.txt","a") as precs:
                    precs.write(f"{each_gram} order accuracy {acc} precision {precision} recall {rec} f1score {f1_score}")
                    precs.write("\n")
            except:
                final_result = {f'{each_gram}-gram_nltk':[0,0,0,0]}

        
        return final_result
    
    import time

    def multiple_train_time_metrics(self, list_ngrams, result_path, test_data, model_name, train_data,proj_number):
        final_result = {}
        log_file = f"{result_path}logs/trained_data_prec_rec_acc_{proj_number}_projects.txt"
        log_file_error = f"{result_path}logs/trained_data_prec_rec_acc_{proj_number}_projects_error.txt"
        real_model_name = f"{result_path}{model_name}"
        for each_gram in list_ngrams:
            for each_run in range(1,6):
                try:
                    # Log the time for training the model
                    train_start_time = time.time()  # Start time
                    
                    self.train_mle(train_data, each_gram, real_model_name)
                    train_end_time = time.time()  # End time
                    train_duration = train_end_time - train_start_time  # Calculate duration
            
                    # Log the time for evaluating the model
                    eval_start_time = time.time()  # Start time

                    acc, precision, rec, f1_score = self.scratch_evaluate_model_nltk_in_order(test_data, f'{real_model_name}_{each_gram}.pkl',result_path,proj_number,each_gram,each_run)
                    eval_end_time = time.time()  # End time
                    eval_duration = eval_end_time - eval_start_time  # Calculate duration

                    # Store results
                    final_result[f'{each_gram}-gram_nltk'] = [acc, precision, rec, f1_score]

                    #Log training and evaluation results, including time
                    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
                        with open(log_file,"a") as fl:
                            fl.write(f"ngram,run,accuracy,precision,recall,f1score,training_time,evaluation_time\n")

                    with open(log_file, "a") as precs:
                        precs.write(f"{each_gram},{each_run},{acc},{precision},{rec},{f1_score},{train_duration},{eval_duration}\n")
        
                except Exception as e:
                    # In case of an exception, log the error and mark the result as 0
                    final_result[f'{each_gram}-gram_nltk'] = [0, 0, 0, 0]
                    with open(log_file_error, "a") as precs:
                        precs.write(f"Error training or evaluating {each_gram}-gram model: {e}\n")
    
        return final_result



    def multiple_train_time_metrics_new(self, train_path, test_path, log_path, model_path, model_number,model_name):
        time_log_file = f"{log_path}/time_logs/time_{model_number}.txt"
        header_check = not os.path.exists(time_log_file) or os.path.getsize(time_log_file) == 0

        if header_check:
                with open(time_log_file, "a") as tm_file:
                    tm_file.write(f"train_data,test_data,train_time,eval_time\n")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(time_log_file), exist_ok=True)

        # Ensure the model_path directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # excluded_20_train = [f"{train_path}/scratch_train_set_20_2_1_proc.txt",f"{train_path}/scratch_train_set_20_2_2_proc.txt",f"{train_path}/scratch_train_set_20_2_3_proc.txt",f"{train_path}/scratch_train_set_20_2_4_proc.txt",f"{train_path}/scratch_train_set_20_2_5_proc.txt",f"{train_path}/scratch_train_set_20_3_1_proc.txt",f"{train_path}/scratch_train_set_20_3_2_proc.txt",f"{train_path}/scratch_train_set_20_3_3_proc.txt",f"{train_path}/scratch_train_set_20_3_4_proc.txt"]
        # excluded_30_train = [f"{train_path}/scratch_train_set_30_2_1_proc.txt",f"{train_path}/scratch_train_set_30_2_2_proc.txt",f"{train_path}/scratch_train_set_30_2_3_proc.txt",f"{train_path}/scratch_train_set_30_2_4_proc.txt",f"{train_path}/scratch_train_set_30_2_5_proc.txt"]
        # excluded_50_train = [f"{train_path}/scratch_train_set_50_2_1_proc.txt",f"{train_path}/scratch_train_set_50_2_2_proc.txt",f"{train_path}/scratch_train_set_50_2_3_proc.txt",f"{train_path}/scratch_train_set_50_2_4_proc.txt"]
        # excluded_80_train = [f"{train_path}/scratch_train_set_80_2_1_proc.txt",f"{train_path}/scratch_train_set_80_2_2_proc.txt",f"{train_path}/scratch_train_set_80_2_3_proc.txt"]

        for each_gram, run in product(range(2, 7), range(1, 6)):
            train_data = f"{train_path}/scratch_train_set_{model_number}_{each_gram}_{run}_proc.txt"
            test_data = f"{test_path}/scratch_test_set_{model_number}_{each_gram}_{run}_proc.txt" 
            if not train_data or not test_data:
                continue
            
        #     if model_number == "20" and len(excluded_20_train) == 9 and train_data in excluded_20_train:
        #         continue


        #     if model_number == "30" and len(excluded_30_train) == 5 and train_data in excluded_30_train:
        #         continue

        #     if model_number == "50" and len(excluded_50_train) == 4 and train_data in excluded_50_train:
        #         continue

        #     if model_number == "80" and len(excluded_80_train) == 3 and train_data in excluded_80_train:
        #         continue
            


            try:
                train_start_time = time.time()
                print(f"training {train_data}")

                model_file = f"{model_path}/{model_name}{model_number}_{each_gram}_{run}.pkl"
                if not os.path.exists(model_file):
                    self.train_mle_new(train_data, each_gram, model_name,model_path,model_number,run)
                    
                
                else:
                    print(f"model {model_file} already trained")
                    
                train_time_duration = time.time() - train_start_time
                # Ensure model_path exists before saving the .pkl file
                

                eval_start_time = time.time()
                print(f"evaluating {test_data}")
                
                self.scratch_evaluate_model_nltk_in_order_all_new_opt(test_data, model_name, log_path,model_path,run,each_gram,model_number)
                eval_time_duration = time.time() - eval_start_time
                
                with open(time_log_file, "a") as tp:
                    tp.write(f"{train_data},{test_data},{train_time_duration},{eval_time_duration}\n")

            except Exception as e:
                print(f"Error: {e}")

    # Get currently occupied CPU cores
    def get_used_cores(self):
        """ Returns a set of currently occupied CPU cores. """
        used_cores = set()
        for proc in psutil.process_iter(attrs=['pid', 'cpu_num']):
            try:
                cpu_core = proc.info['cpu_num']  # Get the core number the process is running on
                if cpu_core is not None:
                    used_cores.add(cpu_core)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return used_cores
    
    def get_available_cores(self,threshold=15):
        """
        Returns a list of CPU cores (by index) with current usage below the threshold.
        The function checks usage over a 1-second interval.
        """
        # psutil.cpu_percent(interval=1, percpu=True) waits 1 sec and returns per-core usage
        usage_per_core = psutil.cpu_percent(interval=1, percpu=True)
        available = [i for i, usage in enumerate(usage_per_core) if usage < threshold]
        print(f"Per-core usage: {usage_per_core} => Available (usage < {threshold}%): {available}")
        return available

    def run_on_core(self,train_path, test_path, log_path, model_path, model_number, model_name, core):
        """
        Sets the CPU affinity for the current process to the given core, then runs the training function.
        """
        # Set affinity so that this process runs only on the specified core.
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity([core])
        print(f"[PID {os.getpid()}] Assigned to core {core}")
        
        # Create an instance of your training class and run the function.
        
        self.multiple_train_time_metrics_new(train_path, test_path, log_path, model_path, model_number, model_name)


    
    def pause_a_process(self,processid):
        try:
            process = psutil.Process(processid)
            process.suspend()
            print(f"Process {processid} has being suspended")
        except psutil.NoSuchProcess:
            print(f"No process with PID {processid} found")
        except psutil.AccessDenied:
            print(f"Access denied. Could not suspend process {processid}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def scratch_evaluate_model_small(self, test_data, model_name, result_path,model_path,run,n,model_number):


        log_file = f"{result_path}/nltk_investigate_{model_number}_{n}_{run}_logs.txt"
        formed_model = f"{model_path}/{model_name}{model_number}_{n}_{run}.pkl"
        file_needs_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0

        with open(test_data, "r", encoding="utf-8") as f, open(log_file, "a") as precs:
            if file_needs_header:
                precs.write("query,expected,answer,rank,correct\n")

            for line in f:
                sentence_tokens = line.strip().split()
                if len(sentence_tokens) < 2:
                    continue  # Skip empty or single-word lines

                for idx in range(1, len(sentence_tokens)):
                    context = ' '.join(sentence_tokens[:idx])
                    true_next_word = sentence_tokens[idx]

                    

                    predicted_next_word, top_10_tokens = self.predict_next_scratch_token_upd_opt(formed_model, context)
                    rank = self.check_available_rank_opt(top_10_tokens, true_next_word)

                    precs.write(f"{context},{true_next_word},{predicted_next_word},{rank},{1 if true_next_word == predicted_next_word else 0}\n")





def main():
    tr_scr = scratch_train_mle()
    # List of datasets, each is a tuple of arguments for multiple_train_time_metrics_new.
    # Define datasets to be processed on separate cores
    datasets = [
        ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs2/10",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models2/10",
        "10", "nltk_"),

        ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs2/20",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models2/20",
        "20", "nltk_"),

        ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs2/30",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models2/30",
        "30", "nltk_"),

        ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs2/50",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models2/50",
        "50", "nltk_"),

        ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs2/80",
        "/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models2/80",
        "80", "nltk_"),
    ]

    processes = []

    for i, data in enumerate(datasets):
        # Continuously check for a core with less than 50% usage.
        available_cores = tr_scr.get_available_cores(threshold=50)
        if not available_cores:
            print("No cores below 50% usage! Waiting...")
            while not available_cores:
                time.sleep(1)
                available_cores = tr_scr.get_available_cores(threshold=50)
        
        # Choose the first available core.
        chosen_core = available_cores[0]
        print(f"Assigning dataset {i+1} to core {chosen_core}")

        # Start a new process with the chosen core.
        p = multiprocessing.Process(target=tr_scr.run_on_core, args=(*data, chosen_core))
        p.start()
        processes.append(p)

    # Wait for all processes to finish.
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
    
# tr_scr = scratch_train_mle()
# tr_scr.multiple_train_time_metrics_new("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/logs/10","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/nltk/models/10","10","nltk_")
#accuracy = tr_scr.paired_t_test([0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407],[0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328])
#print("accuracy parametric t-test result for nltk model ", accuracy)
#precision =tr_scr.paired_t_test([0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743],[0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372])
#print("precision parametric t-test for nltk model",precision)
#f1 = tr_scr.paired_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
#print("f1 parametric ttest for nltk model",f1)
#accuracy_wilcoxon = tr_scr.wilcon_t_test([0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037],[0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037])
#print("accuracy wilcoxon result for nltk model ", accuracy_wilcoxon)
#precision_wilcoxon =tr_scr.wilcon_t_test([0.00039637450273575164,0.001006502629642402,0.01410223779838277,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145],[0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145])
#print("precision parametric wilcoxon for nltk model",precision_wilcoxon)
#f1_wilcoxon = tr_scr.wilcon_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
#print("f1 parametric ttest for nltk model",f1_wilcoxon)
#tr_scr.scratch_evaluate_model_nltk_in_order_all_upd_norun("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_22_projects_model_test_kenlm.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order",10,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/new_metrics")
#tr_scr.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf150_order/scratch_trained_model_nltk_150_projects_2.pkl")
#tr_scr.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/scratch_trained_model_nltk_10_projects_3.pkl")
#tr_scr.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/scratch_trained_model_nltk_10_projects_4.pkl")
#tr_scr.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/scratch_trained_model_nltk_10_projects_5.pkl")
#tr_scr.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/scratch_trained_model_nltk_10_projects_6.pkl")
#accuracy_wilcoxon_2 = tr_scr.wilcon_t_test([0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367,0.24541062801932367],[0.24589371980676328])
#print("accuracy wilcoxon result for nltk model 7 - 11 vs 12 - 16 ", accuracy_wilcoxon_2)
#precision_wilcoxon_2 =tr_scr.wilcon_t_test([0.22551918224779507,0.2264853658226743,0.2264853658226743,0.22696845761011392,0.22696845761011392],[0.22721000350383372])
#print("precision parametric t-test for nltk model 7 - 11 vs 12 - 16 ",precision_wilcoxon_2)
#f1_wilcoxon_2 = tr_scr.wilcon_t_test([0.0006595641494970354,0.0012696922764036857,0.01547662029383393,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823],[0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823])
#print("f1 parametric wilcoxon test for nltk model ",f1_wilcoxon_2)
#tr_scr.multiple_train_time_metrics([2,3,4,5,6],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/models_experiment/scratch_trained_model_nltk_10_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt")
#tr_scr.scratch_evaluate_model_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results/scratch_trained_model_nltk_10_projects_6.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results","10")
#tr_scr.multiple_train_time_metrics([2,3,4,5,6],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","scratch_trained_model_nltk_10_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt","10")
#tr_scr.scratch_evaluate_model_nltk_in_order_all("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10/","10","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10_order/")
#tr_scr.multiple_train_time_metrics([2,3,4,5,6],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/models_experiment/scratch_trained_model_nltk_100_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_100_projects.txt","100")
#tr_scr.multiple_train_time_metrics([2,3,4,5,6],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/models_experiment/scratch_trained_model_nltk_150_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_150_projects.txt","150")
#tr_scr.multiple_train_time_metrics([2,3,4,5,6],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/models_experiment/scratch_trained_model_nltk_500_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_500_projects.txt","500")
#tr_scr.evaluate_mrr_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10/scratch_trained_model_nltk_10_projects_6.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10/","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/testfiles_split","10")
#tr_scr.train_mle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90.txt",8,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2")
#tr_scr.load_trained_model("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_7.pkl")
#tr_scr.scratch_evaluate_model_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_8.pkl") 
#tr_scr.plot_precision_recall_curve("nltk_evaluation_metrics_results_main_portion")
#tr_scr.scratch_evaluate_model_nltk()

# shuffle and split training and test set into four parts for training in parallel
#shuf /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80.txt | split -d -n r/4 - /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_ --additional-suffix=.txt
#shuf /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt | split -d -n r/4 - /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_ --additional-suffix=.txt

#split text file into chunks of 3500 lines per file for evaluating mrr
#split -l 3500 /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_kenlm.txt chunk_ && n=1 && for file in chunk_*; do mv "$file" "scratch_test_data_chunk_$((n++)).txt"; done && read -p "Enter destination folder: " folder && mkdir -p "$folder" && mv scratch_test_data_chunk_*.txt "$folder"
#mrr = 0.2133 for 10 projects

#awk '/Total Reciprocal Rank:/ {rr+=$4} /Total Lines:/ {lines+=$3} END {print "Mean Reciprocal Rank: " rr/lines > "/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/log_path_10/mrr.txt"}' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/log_path_10/kenlm_rr_results_10.txt