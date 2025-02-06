import os
import kenlm
import sys
import nltk
import numpy as np
import subprocess
import random
import scipy.stats as stats
import time
import re
import pandas as pd
import psutil
from multiprocessing import cpu_count,Pool

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_curve,f1_score,confusion_matrix, classification_report
import heapq
from random import sample
import seaborn as sns

class kenlm_train:

    def __init__(self):
        self.result = []
        self.tokenized_data = ""

    def tokenize_kenlm(self,train_data):
        for line in train_data:
            for sentence in nltk.sent_tokenize(line):
                token_sentlist = nltk.word_tokenize(sentence)
                self.tokenized_data += ''.join(token_sentlist).lower()
        return self.tokenized_data

    
    def access_train_data_kenlm(self,file_path,cwd):
        if os.path.isfile(file_path):
            with open(file_path,"r") as each_sentence:
                each_line = each_sentence.readlines()
                val = self.tokenize_kenlm(each_line)
                print(val)
                #module_train = subprocess.run(['/mnt/c/Users/USER/Documents/model_train/online/kenlm/build/bin/lmplz -o 3 > kenlm.arpa'],stdin=val,stdout=subprocess.PIPE, cwd=cwd, shell=True)


    def test_kenlm(self,arpa_file):
        model = kenlm.Model(arpa_file)
        print(model.score("event_whenflagclicked",bos=True,eos=True))
        return model
    
    def replace_non_vs_string_with_tokens(self,string_val):
        if isinstance(string_val,str) and len(string_val) > 0:
            val2 = string_val.split()
            print("see tokens" , val2)
            new_list = ['<S>' if word not in self.valid_opcodes and word not in self.valid_other_field_codes  else word for word in val2  ]
            print("replaced tokens" , new_list)
            return " ".join(new_list)
        else:
            return ""

    
    def scratch_evaluate_model_kenlm(self,test_data,vocab_name,model_name):
        
        model_rec = None
        
        y_true = []
        i=0
        y_pred = []
        
        model_rec = kenlm.Model(model_name)

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            
            for line in lines:
                #line = self.replace_non_vs_string_with_tokens(line)
                line = line.strip()
                sentence_tokens = line.split()
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]

                
                predicted_next_word = self.predict_next_token_kenlm(model_rec,context,vocab_name)
                
                
                i+=1
                if i%500 == 0:
                    print(f"progress {i} true next word {true_next_word} predicted next word {predicted_next_word}")
            
                y_true.append(true_next_word)
                y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        f1score = f1_score(y_true,y_pred,average="weighted")
        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_acc_prec_rec_f1_new_vocab2.txt","a") as frp:
            frp.write(f"vocab name {vocab_name} order accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}\n")
            
        return accuracy,precision,recall,f1score
    

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
        with open(f"{result_path}/tp_fp_fn_tn_label_val_{ngram}_{run}.txt", "w") as af:
            af.write("Class,TP,FP,FN,TN\n")  # Header
            for label, values in metrics.items():
                #print(f"Label {label}: TP={values['TP']}, FP={values['FP']}, FN={values['FN']}, TN={values['TN']}")
                af.write(f"{label},{values['TP']},{values['FP']},{values['FN']},{values['TN']}\n")

        # Print total metrics
        with open(f"{result_path}/total_results_bilstm_tp_tn_fp_fn_{ngram}_{run}.txt","w") as tot:
          tot.write("total_tn,total_fp,total_fn,total_tp\n")
          tot.write(f"{total_tn},{total_fp},{total_fn},{total_tp}")
        print(f"\nTotal TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
        print(f"Confusion Matrix:\n{conf_matrix}")

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
        plt.savefig(f"{result_path}/confusion_matrix_run_an_kenlm_{proj_number}_{ngram}_{run}.pdf")
        plt.close()

    def scratch_evaluate_model_kenlm_time_metrics_in_order(self, test_data, vocab_name, model_name, result_path, proj_number):
        model_rec = kenlm.Model(model_name)
    
        y_true = []
        y_pred = []
        i = 0

        # Start the evaluation timer
        start_time = time.time()

        # Read and shuffle test data
        for each_run in range(1, 2):
            with open(test_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.shuffle(lines)
        
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    sentence_tokens = line.split()
                    if len(sentence_tokens) < 2:
                        continue  # Skip sentences too short to predict

                    # Predict each token in sequence (starting from the second token)
                    for idx in range(1, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:idx])  # Use all preceding tokens as context
                        true_next_word = sentence_tokens[idx]

                        # Predict the next word
                        predicted_next_word = self.predict_next_token_kenlm(model_rec, context, vocab_name)
                    
                        i += 1
                        if i % 500 == 0:
                            print(f"Progress: {i} true next word: {true_next_word} predicted next words: {predicted_next_word}")
                    
                        y_true.append(true_next_word)
                        y_pred.append(predicted_next_word)

            # End the evaluation timer
            end_time = time.time()
            evaluation_time = end_time - start_time

            # Compute the metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
            print(f"Evaluation Time: {evaluation_time}s")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1score}")

            # Save results if needed
            with open(result_path, "a", encoding="utf-8") as result_file:
                result_file.write(f"Project {proj_number}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1score}, Evaluation Time={evaluation_time}s\n")


    def scratch_evaluate_model_kenlm_time_metrics(self, test_data, vocab_name, model_name,result_path,proj_number):
        model_rec = kenlm.Model(model_name)
    
        y_true = []
        y_pred = []
        i = 0

        # Start the evaluation timer
        start_time = time.time()

        # Read and shuffle test data
        for each_run in range(1,2):
            with open(test_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.shuffle(lines)
        
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    sentence_tokens = line.split()
                    if len(sentence_tokens) < 2:
                        continue  # Skip sentences too short to predict

                    context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                    true_next_word = sentence_tokens[-1]

                    # Predict the next word
                    predicted_next_word = self.predict_next_token_kenlm(model_rec, context, vocab_name)
            
                    i += 1
                    if i % 500 == 0:
                        print(f"Progress: {i} true next word: {true_next_word} predicted next words: {predicted_next_word}")
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)

            # End the evaluation timer
            end_time = time.time()
            evaluation_time = end_time - start_time

            # Compute the metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1score = f1_score(y_true, y_pred, average="macro",zero_division=0)
            self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,2,each_run)
            # Log the evaluation metrics and time
            #with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/logs/kenlnm_acc_prec_rec_f1_50_projects.txt", "a") as frp:
                #frp.write(f"Run {each_run} for 50 projects Vocabs name: {vocab_name} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1-score: {f1score} | Evaluation time: {evaluation_time:.2f} seconds\n")

    

    def evaluate_all_models_in_folder(self, test_data, vocab_folder, model_folder,proj_number,new_log_path):
        # Get vocab and model files
        vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab")])
        model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa")])
    
        # Match vocab and model files by order number
        vocab_model_pairs = []
        for vocab in vocab_files:
            vocab_order = vocab.split("order")[1].split(".")[0]
            for model in model_files:
                model_order = model.split("order")[1].split(".")[0]
                if vocab_order == model_order:
                    vocab_model_pairs.append((vocab, model))
                    break

        # Evaluate each vocab-model pair
        for vocab_name, model_name in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            model_path = os.path.join(model_folder, model_name)
            print(f"model  {model_path} vocab {vocab_path}")
            # Load the language model
            model_rec = kenlm.Model(model_path)
            y_true, y_pred = [], []
        
            start_time = time.time()
        
            # Perform evaluation for each run
            for each_run in range(1, 6):
                with open(test_data, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    random.shuffle(lines)

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        sentence_tokens = line.split()
                        if len(sentence_tokens) < 2:
                            continue

                        context = ' '.join(sentence_tokens[:-1])
                        true_next_word = sentence_tokens[-1]
                        predicted_next_word = self.predict_next_token_kenlm(model_rec, context, vocab_path)
                    
                        y_true.append(true_next_word)
                        y_pred.append(predicted_next_word)

                end_time = time.time()
                evaluation_time = end_time - start_time

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
                f1score = f1_score(y_true, y_pred, average="macro",zero_division=0)

                # Log results
                log_path = f"{new_log_path}/metrics_kenlm_{proj_number}.txt"
                if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                    with open(log_path,"a") as fp:
                        fp.write(f"run,vocab_file,model_name,accuracy,precision,recall,f1score,evaluation_time \n")
                with open(log_path, "a") as log_file:
                    log_file.write(f"{each_run},{vocab_name},{model_name},{accuracy},{precision},{recall},{f1score},{evaluation_time:.2f}\n")

    def evaluate_all_models_in_folder_in_order(self, test_data, vocab_folder, model_folder,proj_number,new_log_path):
        # Get vocab and model files
        vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab")])
        model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa")])
    
        # Match vocab and model files by order number
        vocab_model_pairs = []
        for vocab in vocab_files:
            vocab_order = vocab.split("order")[1].split(".")[0]
            for model in model_files:
                model_order = model.split("order")[1].split(".")[0]
                if vocab_order == model_order:
                    vocab_model_pairs.append((vocab, model))
                    break

        # Evaluate each vocab-model pair
        for vocab_name, model_name in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            match =  re.search(r"order(\d+)", vocab_path)
            model_path = os.path.join(model_folder, model_name)
            print(f"model  {model_path} vocab {vocab_path}")
            # Load the language model
            model_rec = kenlm.Model(model_path)
            y_true, y_pred = [], []
        
            start_time = time.time()
        
            # Perform evaluation for each run
            for each_run in range(1, 6):
                with open(test_data, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    random.shuffle(lines)

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        sentence_tokens = line.split()
                        if len(sentence_tokens) < 2:
                            continue
                        
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])
                            true_next_word = sentence_tokens[idx]
                            predicted_next_word = self.predict_next_token_kenlm(model_rec, context, vocab_path)
                            y_true.append(true_next_word)
                            y_pred.append(predicted_next_word)

                end_time = time.time()
                evaluation_time = end_time - start_time

                # Calculate metrics
                # accuracy = accuracy_score(y_true, y_pred)
                # precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                # recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
                # f1score = f1_score(y_true, y_pred, average="macro",zero_division=0)
                
                # Log results
                log_path = f"{new_log_path}/metrics_kenlm_{proj_number}.txt"
                if match:
                    ngram = match.group(1)
                    self.compute_confusion_matrix(y_true,y_pred,new_log_path,proj_number,ngram,each_run)
                
                # if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                #     with open(log_path,"a") as fp:
                #         fp.write(f"run,vocab_file,model_name,accuracy,precision,recall,f1score,evaluation_time \n")
                # with open(log_path, "a") as log_file:
                #     log_file.write(f"{each_run},{vocab_name},{model_name},{accuracy},{precision},{recall},{f1score},{evaluation_time:.2f}\n")

    def check_available_rank(self,list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank
    
    def evaluate_all_models_in_folder_in_order_upd_norun(self, test_data, vocab_folder, model_folder,proj_number,new_log_path):
        # Get vocab and model files
        vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab")])
        model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa")])
     

        # Match vocab and model files by order number
        vocab_model_pairs = []
        for vocab in vocab_files:
            vocab_order = vocab.split("order")[1].split(".")[0]
            for model in model_files:
                model_order = model.split("order")[1].split(".")[0]
                if vocab_order == model_order:
                    vocab_model_pairs.append((vocab, model))
                    break

        # Evaluate each vocab-model pair
        for vocab_name, model_name in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            match =  re.search(r"order(\d+)", vocab_path)
            ngram = match.group(1)
            model_path = os.path.join(model_folder, model_name)
            print(f"model  {model_path} vocab {vocab_path}")
            # Load the language model
            model_rec = kenlm.Model(model_path)
            y_true, y_pred = [], []
        
            start_time = time.time()
            with open(test_data, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    random.shuffle(lines)

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        sentence_tokens = line.split()
                        if len(sentence_tokens) < 2:
                            continue
                        
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])
                            true_next_word = sentence_tokens[idx]
                            predicted_next_word,top_10_tokens = self.predict_next_token_kenlm_upd(model_rec, context, vocab_path)
                            rank = self.check_available_rank(top_10_tokens,true_next_word)

                            investig_path = f"{new_log_path}/kenlm_investigate_{proj_number}_{ngram}.txt"
                            if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                                with open(investig_path,"a") as ip:
                                    ip.write(f"query,expected,answer,rank,correct\n")
                            with open(investig_path,"a") as inv_path_file:
                                inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

        
    
  

    def evaluate_all_models_in_folder_in_order_with_runs(self, testdir, vocab_folder, model_folder,new_log_path):
        # Get vocab and model files

        om_voc = ['kenln_10_6_2.vocab','kenln_10_6_3.vocab','kenln_10_6_4.vocab','kenln_10_6_5.vocab']
        om_arp = ['kenln_10_6_2.arpa','kenln_10_6_3.arpa','kenln_10_6_4.arpa','kenln_10_6_5.arpa']
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

        # Evaluate each vocab-model pair
        for vocab_name, model_name, model_number, ngram_order, run_number in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            model_path = os.path.join(model_folder, model_name)
            
            print(f"Evaluating model {model_path} with vocab {vocab_path}")

            # Generate the corresponding test data file path
            test_data = os.path.join(testdir, f"scratch_test_set_{model_number}_{ngram_order}_{run_number}_proc.txt")
        
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
            print(f"time taken is for {model_path} is {diff}")
                

    def evaluate_all_models_in_folder_in_order_upd(self, test_data, vocab_folder, model_folder,proj_number,new_log_path):
        # Get vocab and model files
        vocab_files = sorted([f for f in os.listdir(vocab_folder) if f.endswith(".vocab")])
        model_files = sorted([f for f in os.listdir(model_folder) if f.endswith(".arpa")])
     

        # Match vocab and model files by order number
        vocab_model_pairs = []
        for vocab in vocab_files:
            vocab_order = vocab.split("order")[1].split(".")[0]
            for model in model_files:
                model_order = model.split("order")[1].split(".")[0]
                if vocab_order == model_order:
                    vocab_model_pairs.append((vocab, model))
                    break

        # Evaluate each vocab-model pair
        for vocab_name, model_name in vocab_model_pairs:
            vocab_path = os.path.join(vocab_folder, vocab_name)
            match =  re.search(r"order(\d+)", vocab_path)
            ngram = match.group(1)
            model_path = os.path.join(model_folder, model_name)
            print(f"model  {model_path} vocab {vocab_path}")
            # Load the language model
            model_rec = kenlm.Model(model_path)
            y_true, y_pred = [], []
        
            start_time = time.time()
        
            # Perform evaluation for each run
            for each_run in range(1, 6):
                with open(test_data, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    random.shuffle(lines)

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        sentence_tokens = line.split()
                        if len(sentence_tokens) < 2:
                            continue
                        
                        for idx in range(1,len(sentence_tokens)):
                            context = ' '.join(sentence_tokens[:idx])
                            true_next_word = sentence_tokens[idx]
                            predicted_next_word,top_10_tokens = self.predict_next_token_kenlm_upd(model_rec, context, vocab_path)
                            rank = self.check_available_rank(top_10_tokens,true_next_word)

                            investig_path = f"{new_log_path}/kenlm_investigate_{proj_number}_{ngram}.txt"
                            if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                                with open(investig_path,"a") as ip:
                                    ip.write(f"query,expected,answer,rank,correct,run_number\n")
                            with open(investig_path,"a") as inv_path_file:
                                inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0},{each_run}\n")
                            

                            


                # end_time = time.time()
                # evaluation_time = end_time - start_time

                # Calculate metrics
                # accuracy = accuracy_score(y_true, y_pred)
                # precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                # recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
                # f1score = f1_score(y_true, y_pred, average="macro",zero_division=0)
                
                # Log results
                # log_path = f"{new_log_path}/metrics_kenlm_{proj_number}.txt"
                # if match:
                #     ngram = match.group(1)
                #     self.compute_confusion_matrix(y_true,y_pred,new_log_path,proj_number,ngram,each_run)
                
                # if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                #     with open(log_path,"a") as fp:
                #         fp.write(f"run,vocab_file,model_name,accuracy,precision,recall,f1score,evaluation_time \n")
                # with open(log_path, "a") as log_file:
                #     log_file.write(f"{each_run},{vocab_name},{model_name},{accuracy},{precision},{recall},{f1score},{evaluation_time:.2f}\n")


    def scratch_evaluate_model_kenlm2(self,test_data,vocab_path,arpa_path):
        arpa_names = []
        model_rec = None
        for i in os.listdir(arpa_path):
            if len(i) > 1 and os.path.isfile(f'{arpa_path}/{i}'):
                i = i.strip()
                arpa_names.append(i)
            else:
                continue
        y_true = []
        i=0
        y_pred = []
        for each_model in arpa_names:
            each_model = each_model.strip()
            model_rec = kenlm.Model(each_model)

            with open(test_data,"r",encoding="utf-8") as f:
                lines= f.readlines()
                random.shuffle(lines)
            
            
                for line in lines:
                    #line = self.replace_non_vs_string_with_tokens(line)
                    line = line.strip()
                    sentence_tokens = line.split()
            
                    context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                    true_next_word = sentence_tokens[-1]
            
                    predicted_next_word,vocab_name = self.predict_next_token_kenlm(model_rec,context,vocab_path)
                
                
                    i+=1
                    if i%500 == 0:
                        print(f"progress {i} true next word {true_next_word} predicted next word {predicted_next_word}")
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        f1score = f1_score(y_true,y_pred,average="weighted")
        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_acc_prec_rec_f1_new.txt","a") as frp:
            frp.write(f"vocab name {vocab_name} order accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}\n")
            
        return accuracy,precision,recall,f1score


    def create_vocab(self,vocab_file,arpa_path):
        arpa_names = []
        for i in os.listdir(arpa_path):
            if len(i) > 1 and os.path.isfile(f'{arpa_path}/{i}'):
                i = i.strip()
                arpa_names.append(i)
            else:
                continue
        print(arpa_names)
        for index,arpa_file in enumerate(arpa_names):  
            arpa_file = arpa_file.strip()  
            with open(f"{arpa_path}/{arpa_file}","r",encoding="utf-8") as fr:
                lines = fr.readlines()
                one_grams_seen = False
                i = 0
                for line in lines:
                
                    line=line.strip()
                
                    if "\\1-grams" in line:
                        one_grams_seen = True
                        continue
                    
                    if one_grams_seen:
                        name = arpa_file.split(".arpa")[0]
                        with open(f"{vocab_file}{index}_{name}.vocab","a") as vf:
                            print(line)
                            token = line.split("\t")
                            if len(token) > 1:
                                print(token)
                                vf.write(token[1]+"\n") 
                            else:
                                continue
                    else:
                        continue
                    

    

    def create_vocab_optimized(self, vocab_path, arpa_path):
        arpa_files = [f for f in os.listdir(arpa_path) if f.endswith(".arpa") and os.path.isfile(f'{arpa_path}/{f}')]
    
        print(arpa_files)  # To check the list of ARPA files found
    
        for arpa_file in arpa_files:  
            arpa_file_path = f"{arpa_path}/{arpa_file}"
            vocab_output_name = arpa_file.replace('.arpa', '.vocab')  # Replace the extension to '.vocab'
        
            with open(arpa_file_path, "r", encoding="utf-8") as arpa_f:
                lines = arpa_f.readlines()
                one_grams_seen = False

                for line in lines:
                    line = line.strip()
                
                    if "\\1-grams" in line:
                        one_grams_seen = True  # Mark that we have reached the 1-grams section
                        continue
                
                    if one_grams_seen:
                        tokens = line.split("\t")
                        if len(tokens) > 1:
                            # Writing vocab to the file matching the ARPA order
                            with open(f"{vocab_path}/{vocab_output_name}", "a") as vocab_f:
                                vocab_f.write(tokens[1] + "\n")
                    else:
                        continue

    def compute_token_score(self,model,context,token):
        context_with_token = context + " " + token
        token_score = model.score(context_with_token)
        return token_score
    
    def load_all_vocab(self,vocab_file):
        
        with open(vocab_file, "r", encoding="utf8") as vocab_f:
            all_voc = vocab_f.readlines()
            return all_voc
            
    def check_vocab_size(self,vocab_file,proj_number):

        with open(vocab_file, "r", encoding="utf8") as vocab_f:
            all_voc = vocab_f.readlines()
            # for each_voc in all_voc:
            #     print(each_voc)
            print(f"vocab size for {proj_number} is {len(all_voc)}" )
                   
    def predict_next_token_kenlm(self,model, context,vocab_name):
        
        next_token_probabilities = {}
        
        with open(vocab_name, "r", encoding="utf8") as vocab_f:
                vocabulary = vocab_f.readlines()
                for candidate_word in vocabulary:
                    candidate_word = candidate_word.strip()
                    context_with_candidate = context + " " + candidate_word
                    next_token_probabilities[candidate_word] = model.score(context_with_candidate)

        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)
        return predicted_next_token
    
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
    
    def predict_next_token_kenlm2(self,model, context,vocab_path):
        vocab_names = []
        for i in os.listdir(vocab_path):
            if len(i) > 1 and os.path.isfile(f'{vocab_path}/{i}'):
                i = i.strip()
                vocab_names.append(i)
            else:
                continue
        next_token_probabilities = {}
        for each_vocab in vocab_names:
            each_vocab = each_vocab.strip()
            with open(f"{vocab_path}/{each_vocab}", "r", encoding="utf8") as vocab_f:
                vocabulary = vocab_f.readlines()
                for candidate_word in vocabulary:
                    candidate_word = candidate_word.strip()
                    context_with_candidate = context + " " + candidate_word
                    next_token_probabilities[candidate_word] = model.score(context_with_candidate)

        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)
        return predicted_next_token,each_vocab
    
    def plot_precision_recall_curve(self,plot_name):

        Accuracy = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        Precision = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737]
        Recall = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        F1 = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169]
        Ngrams = [2,3,4,5,6]

        Accuracy2 = [0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004]
        Precision2 = [0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7419880416193364]
        Recall2 = [0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004]
        F1_2 = [0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5352446315786005]
        Ngrams2 = [7,8,9,10,11]
        
        Accuracy3 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision3 = [0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall3 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_3 = [0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815]
        Ngrams3 = [12,13,14,15,16]

        Accuracy4 = [0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision4 = [0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall4 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_4 = [0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815]
        Ngrams4 = [17,18,19,20]

        Accuracy_all = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,
                        0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004,
                        0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,
                        0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision_all = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,
                         0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7419880416193364,
                         0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916,
                         0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall_all = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,
                      0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004,
                      0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,
                      0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_all  = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,
                   0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5352446315786005,
                   0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815,
                   0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815
                   ]
        
        accurracy_score_2_10 = [0.056335163717714055,0.07443563148261134,0.15578604840349808,0.16900549115314217
                               ,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        precision_2_10 = [0.358121314258974,0.33285318756910176,0.589214697331858,0.5936961737776529,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781]
        recall_2_10 = [0.056335163717714055,0.07443563148261134,0.15578604840349808,0.16900549115314217,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        f1_score_2_10 = [0.03620538803538,0.051215288426201,0.1410810665330756,0.1521354021873837,0.1519912821686130,0.151991282168613,0.151991282168613,0.1519912821686130,0.151991282168613]

        accuracy_score_11_19 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328
                                ,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,
                                0.16819198698393328]
        precision_11_19 = [0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781]
        recall_11_19 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        f1_score_11_19 = [0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861]

        #accurracy_score_12_16 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        #precision_12_16 = [0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781]
        #recall_12_16 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        #f1_score_12_16 = [0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613]

        #accurracy_score_17_20 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        #precision_17_20 = [0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781]
        #recall_17_20 = [0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328]
        #f1_score_17_20 = [0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613]
        main_accuracy = [0.5163701599641202,0.46838092390491853,0.4444183414134079,0.4541357879674519,0.44591333319095317,0.4443115562864404,0.4369860965764688,0.40779104286354995,0.3887619332379386]
        main_precision = [0.7604222887232679,0.7406965196347256,0.7550783228901023,0.7296587264633992,0.7297606037154738,0.7297170540839317,0.7295136930570918,0.7286296210329786,0.733519014367716]
        main_recall = [0.5163701599641202,0.46838092390491853,0.4444183414134079,0.4541357879674519,0.44591333319095317,0.4443115562864404,0.4369860965764688,0.40779104286354995,0.3887619332379386]
        main_f1_score = [0.5125859302212626,0.49460469224057596,0.4833296250062442,0.4888788547708773, 0.4840548194069682,0.48309593108594684,0.47866884739599813,0.4602679270147011,0.4480608614341063]
        main_ngram_2_10 = list(range(2,11))

        accuracy_plot_2_8 = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004]
        recall_2_8 = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004]
        accuracy_plot_9_15 = [0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        recall_9_15 = [0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        precision_2_8 = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,0.7420327232576317,0.7419880416193364]
        precision_9_15 = [0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        f1_2_8 = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,0.5355082179384458,0.5352446315786005]
        f1_9_15 = [0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815]
        
        ngrams2_8 = list(range(2,9))
        ngrams9_15 = list(range(9,16))

        Ngrams_all = list(range(2,21))
        ngrams_2_10 = list(range(2,11))
        ngrams_11_19 = list(range(11,20))
        #[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        #fig = plt.figure()
        #axes = fig.add_axes([0,0,1,1])
        #axes.plot(Ngrams_all,Accuracy_all,label = "Accuracy")
        #axes.plot(Ngrams_all,Precision_all,label = "Precision")
        #axes.plot(Ngrams_all,Recall_all,label="Recall")
        #axes.plot(Ngrams_all,F1_all,label="F1")
        #axes.legend(loc ="center right")

        
        plt.plot(main_ngram_2_10, main_precision, label = "Precision")
        plt.plot(main_ngram_2_10, main_recall, label = "Recall")
        plt.plot(main_ngram_2_10, main_f1_score, label = "F1")
        plt.plot(main_ngram_2_10, main_accuracy, label = "Accuracy")
        
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Kenlm Evaluation Metrics vs N-Gram Orders 2 - 10 on whole datasets')
        plt.legend()
        #plt.xlim(0,21)
        #plt.ylim(0,0.79)
        #plt.show()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}_main_metrics.pdf')

    def paired_t_test(self,nltk_2_10,nltk_11_19):
        if isinstance(nltk_2_10,list) and len(nltk_2_10) > 0 and isinstance(nltk_11_19,list) and len(nltk_11_19) > 0:
            test_val = stats.ttest_rel(nltk_2_10,nltk_11_19)
            #print(test_val)
            return test_val
        
    def wilcon_t_test(self,group1,group2):
        return stats.wilcoxon(group1,group2)
    
    def create_dataframe(data1,data2):
        df = pd.DataFrame({'10-projects':data1,'50-projects':data2})
        print(df)


    def evaluate_mrr_kenlm(self,vocab_file,model_name,result_path,split_folder,proj_number):
        # Ensure result path exists
        model_rec = kenlm.Model(model_name)
        os.makedirs(result_path, exist_ok=True)
        all_vocab = self.load_all_vocab(vocab_file) 

        # Process each file in the split folder
        for split_file in sorted(os.listdir(split_folder)):
            split_file_path = os.path.join(split_folder, split_file)
            if not os.path.isfile(split_file_path):
                continue

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
                            #clearprint(f"processing token {token}")
                            token = token.strip()
                            context = context.strip()
                            context_score = self.compute_token_score(model_rec, context, token)
                            if len(heap) < 10:
                                heapq.heappush(heap, (context_score, token))
                            elif context_score > heap[0][0]:
                                heapq.heappushpop(heap, (context_score, token))

                        heap.sort(reverse=True, key=lambda x: x[0])
                        token_ranks = {t: rank + 1 for rank, (score, t) in enumerate(heap)}
                        print(f"token ranks {token_ranks}")
                        # Compute reciprocal rank
                        true_next_word = true_next_word.strip()
                        rank = token_ranks.get(true_next_word, 0)
                        if rank:
                            current_rank = 1 / rank
                            total_cumulative_rr += current_rank
                            print(f"processed line {line} with reciprocal rank {current_rank} and total cummulative {total_cumulative_rr}")
                        total_count += 1
                    

                # Calculate total RR and lines for the file
                time_spent = time.time() - start_time
                result_file = os.path.join(result_path, f"kenlm_rr_results_{proj_number}_order_main.txt")

                with open(result_file, "a") as rf:
                    rf.write(f"File name : {split_file}\n")
                    rf.write(f"Total Reciprocal Rank: {total_cumulative_rr}\n")
                    rf.write(f"Total Lines: {total_count}\n")
                    rf.write(f"Time Spent: {time_spent:.2f} seconds\n")
                

                print(f"Processed {split_file}: RR = {total_cumulative_rr}, Lines = {total_count}")

    
    

kn = kenlm_train()

kn.evaluate_all_models_in_folder_in_order_with_runs("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/10","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/10","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/logs/10")


# kn.create_vocab("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files/kenln_order2.arpa","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab")

# kn.create_vocab_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/20","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/20")
# kn.create_vocab_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/30","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/30")
# kn.create_vocab_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/50","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/50")
# kn.create_vocab_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/80","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/80")



#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/main_arpa
#print(kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa"))
#model_evaluated = kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa")
#kn.scratch_evaluate_model_kenlm_time_metrics("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_50_projects/kenln_order6.vocab","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_50_projects/kenln_order6.arpa")
#kn.plot_precision_recall_curve("kenlm_evaluation_metrics_plot_order_2_10_whole_datasets")
#kn.plot_precision_recall_curve("kenlm_prec_rec_curv_order")
#accuracy = kn.paired_t_test([0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004],[0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609])
#print("accuracy parametric ttest on kenln",accuracy)
#precision = kn.paired_t_test([0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364],[0.7419880416193364,0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7418981809590166,0.7419431944934916,0.7419431944934916])
#print("precision parametric ttest result on kenln",precision)
#f1 = kn.paired_t_test([0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005],[0.5352446315786005,0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5347160132298798,0.5349805637824815,0.5349805637824815])
#print("f1 parametric ttest result on kenlm",f1)
#print(kn.access_train_data_kenlm("scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/online/kenlm/build")) 


#accuracy_wilconsin = kn.wilcon_t_test([0.056335163717714055,0.07443563148261134,0.15578604840349808,0.16900549115314217,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328],[0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328])
#recall_wilconsin = kn.wilcon_t_test([0.056335163717714055,0.07443563148261134,0.15578604840349808,0.16900549115314217,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328],[0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328,0.16819198698393328])
#accuracy_wilconsin = kn.wilcon_t_test([0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004],[0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609])
#print("recall wilconxon test for kenlm ", recall_wilconsin)

#precision_wilconsin = kn.wilcon_t_test([0.358121314258974,0.33285318756910176,0.589214697331858,0.5936961737776529,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781],[0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781,0.5952517614878781])
#print("precision wilconxon test for kenlm ", precision_wilconsin)

#f1_wilconxon = kn.wilcon_t_test([0.03620538803538,0.051215288426201,0.1410810665330756,0.1521354021873837,0.1519912821686130,0.151991282168613,0.151991282168613,0.1519912821686130,0.151991282168613],[0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861,0.151991282168613,0.15199128216861])
#print("f1 wilconxon test for kenlm ", f1_wilconxon)
#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 7  --discount_fallback < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order7.arpa       
#cmake -DKENLM_MAX_ORDER=10 ..
#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 20  --discount_fallback < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order20.arpa
#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 2  < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects_kenlm.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_10_projects_upd/kenln_order2.arpa

#kn.evaluate_all_models_in_folder_in_order_upd_norun("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_22_projects_model_test_kenlm.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_10_projects_upd","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_10_projects_upd","10","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/log_path_10_order/new_metrics2")

# kn.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects_upd/kenln_order2.vocab",500)
# kn.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects_upd/kenln_order3.vocab",500)
# kn.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects_upd/kenln_order4.vocab",500)
# kn.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects_upd/kenln_order5.vocab",500)
# kn.check_vocab_size("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects_upd/kenln_order6.vocab",500)

#kn.scratch_evaluate_model_kenlm_time_metrics("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_kenlm.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_10_projects_upd/kenln_order2.vocab","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_10_projects_upd/kenln_order2.arpa","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/log_path_10","10")
#kn.evaluate_mrr_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_10_projects_upd/kenln_order2.vocab","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_10_projects_upd/kenln_order2.arpa","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/log_path_10/","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/testfiles_split","10")
#kn.evaluate_all_models_in_folder("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_150_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_150_projects","150")
#kn.evaluate_all_models_in_folder("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/vocab_500_projects","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/kenlm/arpa_files_500_projects","500")

#preprocess datasets by replacing _, > and < with UNDERSCORE,RIGHTANG AND LEFTANG and then converting them to lowercase
#sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects_kenlm.txt

#preprocessing my csvfile for kenlm metrics to insert ngram values based on the number in the vocab file
#awk -F',' 'NR==1 {print $1",ngram,"substr($0,index($0,$2))} NR>1 {split($2,a,"order"); print $1","a[2]+0","substr($0,index($0,$2))}' metrics_kenlm_500.csv > metrics_kenlm_500_pro.csv

#awk '/Total Reciprocal Rank:/ {rr+=$NF} /Total Lines:/ {lines+=$NF} END {print rr/lines > "/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10/nltk_mrr_order10.txt"}' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/results_conf10/nltk_rr_results_10_order.txt


# sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_22_projects_model_test.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_22_projects_model_test_kenlm.txt
# sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_44_projects_model_test.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_44_projects_model_test_kenlm.txt
# sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_66_projects_model_test.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_66_projects_model_test_kenlm.txt
# sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_88_projects_model_test.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_88_projects_model_test_kenlm.txt
# sed -e 's/>/RIGHTANG/g' -e 's/</LEFTANG/g' -e 's/_/UNDERSCORE/g' /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_120_projects_model_test.txt | tr '[:upper:]' '[:lower:]' > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_120_projects_model_test_kenlm.txt




