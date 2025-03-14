import pandas as pd
import os
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force TensorFlow to use CPU
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import re
import psutil
from tensorflow.keras.layers import Input
import multiprocessing
import itertools
import gc
import heapq

class bilstm_cybera:
    def consolidate_data_train_parallel(self, train_path, result_path, test_path, model_number, logs_path):
        """
        Spawns a separate process for each run and assigns processes to the two available CPU cores.
        """
        processes = []
        available_cores = self.get_available_cores()  
        while not available_cores:
                print("No cores below 20% usage! Waiting for a free core...")
                time.sleep(1)
                available_cores = self.get_available_cores()

        core_index = 0  # Track which core to assign next

        for each_run in range(1, 6):  # 5 runs
            # Assign 1 core per run, cycling through the available cores
            chosen_core = available_cores[core_index % len(available_cores)] 
            core_index += 1

            print(f"Assigning run {each_run} to core {chosen_core}")

            # Start a new process for this run.
            p = multiprocessing.Process(
                target=self.run_consolidate_train_run,
                args=(train_path, result_path, test_path, model_number, logs_path, each_run, [chosen_core])
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish.
        for p in processes:
            p.join()

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

    
    def eval_five_runs_opt(self, max_seq, result_path, test_data, proj_number, runs, logs_path):
        all_models = sorted([files for files in os.listdir(result_path) if files.endswith(".keras")])
        print(all_models)
        # print(tf.__version__)
        # print("max length", max_seq)

        # # Force TensorFlow to use CPU
        # tf.config.set_visible_devices([], 'GPU')

        # # Check if it's using CPU
        # print("Is TensorFlow using GPU?", len(tf.config.list_physical_devices('GPU')) > 0)

        # # Reduce model complexity to save memory
        # model = Sequential([
        #     Input(shape=(max_seq - 1,)),  # Explicitly define the input shape
        #     Embedding(total_words, 50),  # Reduced embedding dimension from 100 to 50
        #     Bidirectional(LSTM(100)),  # Reduced LSTM units from 150 to 100
        #     Dense(total_words, activation='softmax')
        # ])
        # adam = Adam(learning_rate=0.01)
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # # Use a data generator to reduce memory usage
        # train_generator = self.DataGenerator(xs, ys, batch_size=16)
        # lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        # early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # # Fit the model
        # history = model.fit(train_generator, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

        # # Save the history
        # with open(f"{result_path}main_historyrec_150embedtime_6_{runs}.pickle", "wb") as hs:
        #     pickle.dump(history.history, hs)

        # # Save the model for every run
        # file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_sample_project{proj_number}_6_{runs}.keras"

        # if os.path.exists(file_name):
        #     os.remove(file_name)
        # model.save(file_name)

        # Evaluate the model
        for model in all_models:
            complete_model = f"{result_path}{model}"
            self.evaluate_bilstm_in_order_upd_norun_opt_new_2(test_data, max_seq, complete_model, result_path, proj_number, runs, logs_path)


    def eval_five_runs_opt_main(self, max_seq, result_path, test_data, proj_number, runs, logs_path):
        spec_model = os.path.join(f"{result_path}main_bilstm_scratch_model_150embedtime1_main_sample_project{proj_number}_6_{runs}.keras")
        print(f"model is {spec_model}")
        self.evaluate_bilstm_in_order_upd_norun_opt_new_2(test_data, max_seq, spec_model, result_path, proj_number, runs, logs_path)
            

    def train_model_five_runs_opt(self, total_words, max_seq, xs, ys, result_path, test_data, proj_number, runs, logs_path):
        print(tf.__version__)
        print("max length", max_seq)

        # Force TensorFlow to use CPU
        tf.config.set_visible_devices([], 'GPU')

        # Check if it's using CPU
        print("Is TensorFlow using GPU?", len(tf.config.list_physical_devices('GPU')) > 0)

        # Reduce model complexity to save memory
        model = Sequential([
            Input(shape=(max_seq - 1,)),  # Explicitly define the input shape
            Embedding(total_words, 50),  # Reduced embedding dimension from 100 to 50
            Bidirectional(LSTM(100)),  # Reduced LSTM units from 150 to 100
            Dense(total_words, activation='softmax')
        ])
        adam = Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Use a data generator to reduce memory usage
        train_generator = self.DataGenerator(xs, ys, batch_size=16)
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Fit the model
        history = model.fit(train_generator, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

        # Save the history
        with open(f"{result_path}main_historyrec_150embedtime_6_{runs}.pickle", "wb") as hs:
            pickle.dump(history.history, hs)

        # Save the model for every run
        file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_sample_project{proj_number}_6_{runs}.keras"

        if os.path.exists(file_name):
            os.remove(file_name)
        model.save(file_name)

        # Evaluate the model
        self.evaluate_bilstm_in_order_upd_norun_opt(test_data, max_seq, model, result_path, proj_number, runs, logs_path)

    class DataGenerator(tf.keras.utils.Sequence):
        """
        Data generator to load data in smaller chunks and reduce memory usage.
        """
        def __init__(self, xs, ys, batch_size):
            self.xs = xs
            self.ys = ys
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.xs) / self.batch_size))

        def __getitem__(self, idx):
            batch_x = self.xs[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.ys[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

    def evaluate_bilstm_in_order_upd_norun_opt(self, test_data, maxlen, model, result_path, proj_number, run, logs_path):
        tokenz = None
        with open(f"{result_path}tokenized_file_50embedtime1_{run}.pickle", "rb") as tk:
            tokenz = pickle.load(tk)

        with open(test_data, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)

            for line in lines:
                line = line.strip()
                sentence_tokens = line.split(" ")
                if len(sentence_tokens) < 2:
                    continue

                # Evaluate each token in order starting from the second token
                for idx in range(1, len(sentence_tokens)):
                    context = ' '.join(sentence_tokens[:idx])
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word, top_10_tokens = self.predict_token_score_upd(context, tokenz, model, maxlen)
                    rank = self.check_available_rank(top_10_tokens, true_next_word)
                    investig_path = f"{logs_path}/bilstm_investigate_{proj_number}_6_{run}_logs.txt"
                    if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                        with open(investig_path, "a") as ip:
                            ip.write(f"query,expected,answer,rank,correct\n")
                    with open(investig_path, "a") as inv_path_file:
                        inv_path_file.write(
                            f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")
                        
  
    def evaluate_bilstm_in_order_upd_norun_opt_new_2(self, test_data, maxlen, model, result_path, proj_number, run, logs_path):
        # Load pre-trained model
        
        loaded_model = load_model(model, compile=False)

        # Load tokenized data once
        with open(f"{result_path}tokenized_file_50embedtime1_{run}.pickle", "rb") as tk:
            tokenz = pickle.load(tk)

        # Log file path
        investig_path = f"{logs_path}/bilstm_investigate_{proj_number}_6_{run}_logs.txt"
        if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
            print(f"creating log file {investig_path}")
            with open(investig_path, "w") as log_file:
                log_file.write("query,expected,answer,rank,correct\n")
            
            with open(test_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
                random.shuffle(lines)

                for line in lines:
                    line = line.strip()
                    sentence_tokens = line.split(" ")
                    if len(sentence_tokens) < 2:
                        continue

                    # Evaluate each token in order starting from the second token
                    for idx in range(1, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:idx])
                        true_next_word = sentence_tokens[idx]

                        predicted_next_word, top_10_tokens = self.predict_token_score_upd_opt2(context, tokenz, loaded_model, maxlen)
                        rank = self.check_available_rank(top_10_tokens, true_next_word)
                        
                        with open(investig_path, "a") as inv_path_file:
                            inv_path_file.write(
                                f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

        else:
            # Count the number of existing log entries
            log_entry_count = self.count_log_entries(investig_path)

            # Find resume point
            resume_point = self.find_resume_point(test_data, log_entry_count)
            if resume_point is None:
                print("Evaluation completed")
                return

            line_num, token_pos = resume_point
            print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")

            # Process test data
            with open(test_data, 'r') as test_file, open(investig_path, "a") as inv_path_file:
                # Skip lines until the resume point
                skipped_lines = itertools.islice(test_file, line_num, None)

                for line in skipped_lines:
                    line = line.strip()
                    sentence_tokens = line.split(" ")
                    if len(sentence_tokens) < 2:
                        continue

                    
                     
                    for i in range(token_pos, len(sentence_tokens)):
                        context = ' '.join(sentence_tokens[:i])
                        true_next_word = sentence_tokens[i]

                        predicted_next_word, top_10_tokens = self.predict_token_score_upd_opt(
                                context, tokenz, loaded_model, maxlen
                            )
                        rank = self.check_available_rank(top_10_tokens, true_next_word)

                        # Write log entry
                        inv_path_file.write(
                                f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n"
                            )

                    token_pos = 1  # Reset token position after processing first resumed line
        
        del loaded_model
        gc.collect()

    def evaluate_bilstm_in_order_upd_norun_opt_new(self, test_data, maxlen, model, result_path, proj_number, run, logs_path):
        tokenz = None
        loaded_model = load_model(f"{model}",compile=False)
        # Count the number of log entries already generated
        log_entry_count = self.count_log_entries(f"{logs_path}/bilstm_investigate_{proj_number}_6_{run}_logs.txt")

        resume_point = self.find_resume_point(test_data, log_entry_count)

        with open(f"{result_path}tokenized_file_50embedtime1_{run}.pickle", "rb") as tk:
            tokenz = pickle.load(tk)


        if resume_point is None:
            print("Evaluation completed")
            return
                
        line_num, token_pos = resume_point
        print(f"Resuming evaluation from line {line_num + 1}, token position {token_pos + 1}.")

                

        # Open files for reading and appending
        with open(test_data, 'r') as test_file:
            # Skip lines until the resume point
            for _ in range(line_num):
                next(test_file)

            for line in test_file:
                tokens = line.strip().split()
                if len(tokens) >= 2:  # Only evaluate lines with 2 or more tokens
                    # Skip tokens until the resume point
                    for i in range(token_pos, len(tokens) - 1):
                        context = ' '.join(tokens[:i])
                        true_next_word = tokens[i]
                        predicted_next_word, top_10_tokens = self.predict_token_score_upd_opt2(context, tokenz, loaded_model, maxlen)
                        rank = self.check_available_rank(top_10_tokens, true_next_word)
                                       
                    
                        investig_path = f"{logs_path}/bilstm_investigate_{proj_number}_6_{run}_logs.txt"
                        if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                            with open(investig_path, "a") as ip:
                                ip.write(f"query,expected,answer,rank,correct\n")
                        with open(investig_path, "a") as inv_path_file:
                            inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

                    token_pos = 1  # Reset token position after the first line

    def predict_token_score_upd(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.
        """
        token_list = tokenz.texts_to_sequences([context])
        vocab = list(tokenz.word_index.keys())
        max_prob_tokens = {}

        if not token_list or len(token_list[0]) == 0:
            return -1, []

        for each_token in vocab:
            token_value = token_list[0][-maxlen + 1:] + [tokenz.word_index.get(each_token, 0)]
            padded_in_seq = pad_sequences([token_value], maxlen=maxlen - 1, padding="pre")
            padded_in_seq = tf.convert_to_tensor(padded_in_seq)

            prediction = model.predict(padded_in_seq, verbose=0)[0]
            token_index = tokenz.word_index.get(each_token, 0)
            max_prob_tokens[each_token] = prediction[token_index]

        predicted_next_token = max(max_prob_tokens, key=max_prob_tokens.get)
        top_10_tokens_scores = sorted(max_prob_tokens.items(), key=lambda item: item[1], reverse=True)[:10]

        return predicted_next_token, top_10_tokens_scores
    
    def predict_token_score_upd_opt(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.
        Optimized to reduce redundant computations and improve efficiency.
        """
        # Tokenize the context
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return -1, []

        # Prepare the base sequence (context without the last token)
        base_sequence = token_list[0][-maxlen + 1:]

        # Precompute all token indices
        vocab = list(tokenz.word_index.keys())
        token_indices = [tokenz.word_index.get(token, 0) for token in vocab]

        # Create a batch of sequences for all tokens
        padded_sequences = [
            base_sequence + [token_index] for token_index in token_indices
        ]
        padded_sequences = pad_sequences(padded_sequences, maxlen=maxlen - 1, padding="pre")
        padded_sequences = tf.convert_to_tensor(padded_sequences)

        # Perform batch prediction
        predictions = model.predict(padded_sequences, verbose=0)

        # Extract probabilities for each token
        max_prob_tokens = {
            token: predictions[i][token_index]
            for i, (token, token_index) in enumerate(zip(vocab, token_indices))
        }

        # Find the predicted next token
        predicted_next_token = max(max_prob_tokens, key=max_prob_tokens.get)

        # Find the top-10 tokens without sorting the entire vocabulary
        top_10_tokens_scores = []
        for token, prob in max_prob_tokens.items():
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


    def predict_token_score_upd_opt2(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.
        Optimized to reduce redundant computations and improve efficiency.
        """
        # Tokenize the context
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return -1, []

        # Prepare the base sequence (context without the last token)
        base_sequence = token_list[0][-maxlen + 1:]

        # Precompute all token indices
        vocab = list(tokenz.word_index.keys())
        token_indices = [tokenz.word_index.get(token, 0) for token in vocab]

        # Create a batch of sequences for all tokens
        padded_sequences = [
            base_sequence + [token_index] for token_index in token_indices
        ]
        padded_sequences = pad_sequences(padded_sequences, maxlen=maxlen - 1, padding="pre")
        padded_sequences = tf.convert_to_tensor(padded_sequences)

        # Perform batch prediction
        predictions = model(padded_sequences, training=False)  # Use model.call() for raw logits

        # Extract probabilities for each token
        max_prob_tokens = {
            token: predictions[i][token_index].numpy()
            for i, (token, token_index) in enumerate(zip(vocab, token_indices))
        }

        # Find the predicted next token
        predicted_next_token = max(max_prob_tokens, key=max_prob_tokens.get)

        # Use a min-heap to find the top-10 tokens efficiently
        top_10_tokens_scores = heapq.nlargest(
            10, max_prob_tokens.items(), key=lambda x: x[1]
        )

        return predicted_next_token, top_10_tokens_scores

    def check_available_rank(self, list_tuples, true_word):
        rank = -1

        for ind, val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank

    def tokenize_data_inp_seq(self, file_name, result_path, run, chunk_size=100000):
        self.tokenizer = Tokenizer(oov_token='<oov>')
        self.encompass = []

        with open(file_name, "r", encoding="utf-8") as rf:
            while True:
                lines = rf.readlines(chunk_size)
                if not lines:
                    break

                # Fit the tokenizer on the chunk
                self.tokenizer.fit_on_texts(lines)

                # Process each line in the chunk
                for each_line in lines:
                    each_line = each_line.strip()
                    self.token_list = self.tokenizer.texts_to_sequences([each_line])[0]
                    for i in range(1, len(self.token_list)):
                        ngram_seq = self.token_list[:i + 1]
                        self.encompass.append(ngram_seq)

        # Save the tokenizer
        with open(f"{result_path}tokenized_file_50embedtime1_{run}.pickle", "wb") as tk:
            pickle.dump(self.tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

        self.total_words = len(self.tokenizer.word_index) + 1
        print(f"Total words (vocabulary size): {self.total_words}")

        return self.encompass, self.total_words, self.tokenizer

    def pad_sequ(self, input_seq):
        max_seq_len = max([len(x) for x in input_seq])
        padded_in_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding='pre'))
        return padded_in_seq, max_seq_len

    def prep_seq_labels(self, padded_seq, total_words):
        xs, labels = padded_seq[:, :-1], padded_seq[:, -1]

        max_label_index = np.max(labels)
        if max_label_index >= total_words:
            print(f"Adjusting total_words from {total_words} to {max_label_index + 1} based on labels.")
            total_words = max_label_index + 1

        if np.any(labels >= total_words):
            raise ValueError(f"Labels contain indices >= total_words: {np.max(labels)} >= {total_words}")

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        return xs, ys, labels

    def run_consolidate_train_run(self, train_path, result_path, test_path, model_number, logs_path, each_run, cores):
        """
        Sets CPU affinity for this process to the chosen cores and performs one run of training.
        """
        # Set the CPU affinity so this process runs on the designated cores.
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cores)
        print(f"[PID {os.getpid()}] Running run {each_run} on cores {cores}")

        # Construct file paths.
        #train_data = f"{train_path}/scratch_train_set_{model_number}_6_{each_run}_proc.txt"
        test_data = f"{test_path}/scratch_test_set_{model_number}_6_{each_run}_proc.txt"

        # # Run your sequence of operations.
        # input_seq, total_words, tokenizer = self.tokenize_data_inp_seq(train_data, result_path, each_run)
        # padd_seq, max_len = self.pad_sequ(input_seq)
        # xs, ys, labels = self.prep_seq_labels(padd_seq, total_words)
        # print(f"Maximum length for run {each_run}: {max_len}")
        self.eval_five_runs_opt_main(47,result_path,test_data,model_number,each_run,logs_path)

        #self.train_model_five_runs_opt(total_words, max_len, xs, ys, result_path, test_data, model_number, each_run, logs_path)

    def get_available_cores(self, threshold=20, num_cores=1):
        """
        Returns a list of CPU core indices whose usage is below the given threshold.
        """
        usage_per_core = psutil.cpu_percent(interval=1, percpu=True)
        available = [i for i, usage in enumerate(usage_per_core) if usage < threshold]
        print(f"Per-core usage: {usage_per_core} => Available (usage < {threshold}%): {available}")
        #return available[:num_cores]
        return available

    def pin_process_to_cores(self, cores):
        """
        Pins the current process to the specified CPU cores.
        """
        p = psutil.Process(os.getpid())
        p.cpu_affinity(cores)

# Example usage
cl_ob = bilstm_cybera()





sample = ("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_train","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/bilstm/20/","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/output_test",20,"/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/bilstm/20/logs")
cl_ob.consolidate_data_train_parallel(*sample)