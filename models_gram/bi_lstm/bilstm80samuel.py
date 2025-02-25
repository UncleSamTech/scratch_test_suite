import pandas as pd
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import re
import psutil
import multiprocessing
import gc


class bilstm_cybera:
    def __init__(self):
        self.total_words = 0
        self.tokenizer = None

    def consolidate_data_train_parallel(self, train_path, result_path, test_path, model_number, logs_path):
        """
        Spawns a separate process for each run, assigning processes to available CPU cores.
        """
        processes = []
        for each_run in range(1, 2):  # Single run for simplicity; adjust to 5 if needed
            chosen_core = [each_run % multiprocessing.cpu_count()]
            print(f"Assigning run {each_run} to core {chosen_core}")
            p = multiprocessing.Process(
                target=self.run_consolidate_train_run,
                args=(train_path, result_path, test_path, model_number, logs_path, each_run, chosen_core)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def train_model_five_runs_opt(self, total_words, max_seq, xs, ys, result_path, test_data, proj_number, runs, logs_path):
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Maximum sequence length: {max_seq}")

        # Simplified model
        model = Sequential([
            Input(shape=(max_seq - 1,)),
            Embedding(total_words, 50),  # Reduced embedding size
            Bidirectional(LSTM(100)),    # Reduced LSTM units
            Dense(total_words, activation='softmax')
        ])
        adam = Adam(learning_rate=0.01)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        train_generator = self.DataGenerator(xs, ys, batch_size=8, shuffle=True)  # Added shuffling
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(train_generator, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

        # Save training history
        with open(os.path.join(result_path, f"main_historyrec_50embedtime_6_{runs}.pickle"), "wb") as hs:
            pickle.dump(history.history, hs)

        # Save model
        file_name = os.path.join(result_path, f"main_bilstm_scratch_model_50embedtime1_main_sample_project{proj_number}_6_{runs}.keras")
        model.save(file_name)

        # Clear memory
        K.clear_session()
        del model, train_generator, history
        gc.collect()

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, xs, ys, batch_size, shuffle=True):
            self.xs = xs
            self.ys = ys
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.xs) / self.batch_size))

        def __getitem__(self, idx):
            batch_x = self.xs[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.ys[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

        def on_epoch_end(self):
            if self.shuffle:
                indices = np.arange(len(self.xs))
                np.random.shuffle(indices)
                self.xs = self.xs[indices]
                self.ys = self.ys[indices]

    def evaluate_bilstm_in_order_upd_norun_opt(self, test_data, maxlen, model, result_path, proj_number, run, logs_path):
        tokenz = None
        with open(os.path.join(result_path, f"tokenized_file_50embedtime1_{run}.pickle"), "rb") as tk:
            tokenz = pickle.load(tk)

        with open(test_data, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)

            for line in lines:
                line = line.strip()
                sentence_tokens = line.split(" ")
                if len(sentence_tokens) < 2:
                    continue

                for idx in range(1, len(sentence_tokens)):
                    context = ' '.join(sentence_tokens[:idx])
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word, top_10_tokens = self.predict_token_score_upd(context, tokenz, model, maxlen)
                    rank = self.check_available_rank(top_10_tokens, true_next_word)
                    investig_path = os.path.join(logs_path, f"bilstm_investigate_{proj_number}_6_{run}_logs.txt")
                    if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                        with open(investig_path, "a") as ip:
                            ip.write("query,expected,answer,rank,correct\n")
                    with open(investig_path, "a") as inv_path_file:
                        inv_path_file.write(
                            f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

    def predict_token_score_upd(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.
        """
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return -1, []

        padded_in_seq = pad_sequences(token_list, maxlen=maxlen - 1, padding="pre")
        padded_in_seq = tf.convert_to_tensor(padded_in_seq)

        predictions = model.predict(padded_in_seq, verbose=0)[0]
        top_k_indices = np.argsort(predictions)[-10:][::-1]
        top_k_tokens = [(tokenz.index_word.get(idx, '<oov>'), predictions[idx]) for idx in top_k_indices]

        predicted_next_token = top_k_tokens[0][0]
        return predicted_next_token, top_k_tokens

    def check_available_rank(self, list_tuples, true_word):
        for ind, val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                return ind + 1
        return -1

    def tokenize_data_inp_seq(self, file_name, result_path, run, chunk_size=100000):
        self.tokenizer = Tokenizer(oov_token='<oov>')
        with open(file_name, "r", encoding="utf-8") as rf:
            while True:
                lines = rf.readlines(chunk_size)
                if not lines:
                    break
                self.tokenizer.fit_on_texts(lines)
                chunk_seqs = []
                for each_line in lines:
                    each_line = each_line.strip()
                    token_list = self.tokenizer.texts_to_sequences([each_line])[0]
                    for i in range(1, len(token_list)):
                        ngram_seq = token_list[:i + 1]
                        chunk_seqs.append(ngram_seq)
                yield chunk_seqs  # Yield chunks instead of storing all in memory

        # Save tokenizer after processing all chunks
        with open(os.path.join(result_path, f"tokenized_file_50embedtime1_{run}.pickle"), "wb") as tk:
            pickle.dump(self.tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

        self.total_words = len(self.tokenizer.word_index) + 1
        print(f"Total words (vocabulary size): {self.total_words}")

    def pad_sequ(self, input_seq):
        max_seq_len = max([len(x) for x in input_seq])
        padded_in_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding='pre'))
        return padded_in_seq, max_seq_len

    def prep_seq_labels(self, padded_seq, total_words):
        xs, labels = padded_seq[:, :-1], padded_seq[:, -1]
        if np.max(labels) >= total_words:
            print(f"Warning: Some label indices exceed total_words ({total_words}). Clipping labels.")
            labels = np.clip(labels, 0, total_words - 1)
        return xs, labels

    def run_consolidate_train_run(self, train_path, result_path, test_path, model_number, logs_path, each_run, cores):
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cores)
        print(f"[PID {os.getpid()}] Running run {each_run} on cores {cores}")

        train_data = os.path.join(train_path, f"scratch_train_set_{model_number}_6_{each_run}_proc.txt")
        test_data = os.path.join(test_path, f"scratch_test_set_{model_number}_6_{each_run}_proc.txt")

        # Process data in chunks
        input_seq_gen = self.tokenize_data_inp_seq(train_data, result_path, each_run)
        all_xs, all_ys = [], []
        max_len = 0
        for chunk_seqs in input_seq_gen:
            padd_seq, chunk_max_len = self.pad_sequ(chunk_seqs)
            xs, labels = self.prep_seq_labels(padd_seq, 908)
            all_xs.append(xs)
            all_ys.append(labels)
            max_len = max(max_len, chunk_max_len)
            del padd_seq, xs, labels, chunk_seqs
            gc.collect()

        # Concatenate chunks
        xs = np.concatenate(all_xs, axis=0)
        ys = np.concatenate(all_ys, axis=0)
        print(f"Maximum length for run {each_run}: {max_len}")

        self.train_model_five_runs_opt(908, max_len, xs, ys, result_path, test_data, model_number, each_run, logs_path)

        # Clear memory
        del xs, ys, all_xs, all_ys
        gc.collect()


# Example usage
cl_ob = bilstm_cybera()
sample = (
    "/mnt/siwuchuk/thesis/another/kenlm/output_train",
    "/mnt/siwuchuk/thesis/another/bilstm/models/80/",
    "/mnt/siwuchuk/thesis/another/kenlm/output_test",
    80,
    "/mnt/siwuchuk/thesis/another/bilstm/logs/80"
)
cl_ob.consolidate_data_train_parallel(*sample)