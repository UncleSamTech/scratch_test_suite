import pandas as pd
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pickle
import psutil
import multiprocessing

class BiLSTMCybera:
    def consolidate_data_train_parallel(self, train_path, result_path, test_path, model_number, logs_path):
        """
        Distributes five runs across available CPU cores to optimize memory and processing speed.
        """
        processes = []
        available_cores = self.get_available_cores(threshold=50, num_cores=16)  # Use less-loaded cores
        core_index = 0  # Track assigned core

        for each_run in range(1, 6):  
            if not available_cores:
                available_cores = self.get_available_cores(threshold=50, num_cores=16)  # Refresh core list
            
            chosen_core = available_cores[core_index % len(available_cores)]
            core_index += 1

            print(f"Assigning run {each_run} to core {chosen_core}")
            p = multiprocessing.Process(
                target=self.run_consolidate_train_run,
                args=(train_path, result_path, test_path, model_number, logs_path, each_run, [chosen_core])
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def train_model_five_runs_opt(self, total_words, max_seq, xs, ys, result_path, test_data, proj_number, runs, logs_path):
        print("TensorFlow Version:", tf.__version__)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {tf.test.gpu_device_name()}")
        else:
            print("No GPU available. Running on CPU.")

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        model = Sequential([
            Embedding(total_words, 100, input_shape=(max_seq - 1,)),
            Bidirectional(LSTM(150)),
            Dense(total_words, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

        history = model.fit(xs, ys, epochs=50, batch_size=64, verbose=1, callbacks=[lr_scheduler, early_stopping])

        with open(f"{result_path}train_history_{runs}.pickle", "wb") as hs:
            pickle.dump(history.history, hs)

        model.save(f"{result_path}bilstm_model_{proj_number}_{runs}.keras")

        self.evaluate_bilstm_in_order(test_data, max_seq, model, result_path, proj_number, runs, logs_path)

    def tokenize_data_inp_seq(self, file_name, result_path, run, chunk_size=50000):
        tokenizer = Tokenizer(oov_token='<oov>')
        sequences = []

        with open(file_name, "r", encoding="utf-8") as rf:
            while True:
                lines = rf.readlines(chunk_size)
                if not lines:
                    break

                tokenizer.fit_on_texts(lines)
                for line in lines:
                    token_list = tokenizer.texts_to_sequences([line.strip()])[0]
                    for i in range(1, len(token_list)):
                        sequences.append(token_list[:i + 1])

        with open(f"{result_path}tokenizer_{run}.pickle", "wb") as tk:
            pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

        total_words = len(tokenizer.word_index) + 1
        print(f"Total words: {total_words}")

        return sequences, total_words, tokenizer

    def pad_sequ(self, input_seq, chunk_size=5000):
        if not isinstance(input_seq, list) or not all(isinstance(x, list) for x in input_seq):
            raise TypeError("input_seq must be a list of lists")
        
        max_seq_len = max(len(x) for x in input_seq) if input_seq else 0
        num_samples = len(input_seq)

        # Use memory-mapped array for large datasets
        mmap_path = "padded_sequences.dat"
        padded_in_seq = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=(num_samples, max_seq_len))

        for i in range(0, num_samples, chunk_size):
            chunk = input_seq[i:i + chunk_size]
            for j, seq in enumerate(chunk):
                padded_in_seq[i + j, -len(seq):] = seq  # Pad sequences at the end

        return padded_in_seq, max_seq_len

    def prep_seq_labels(self, padded_seq, total_words):
        xs, labels = padded_seq[:, :-1], padded_seq[:, -1]
        ys = keras.utils.to_categorical(labels, num_classes=total_words)
        return xs, ys, labels

    def run_consolidate_train_run(self, train_path, result_path, test_path, model_number, logs_path, each_run, cores):
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cores)
        print(f"Running run {each_run} on cores {cores}")

        train_data = f"{train_path}/train_{model_number}_{each_run}.txt"
        test_data = f"{test_path}/test_{model_number}_{each_run}.txt"

        input_seq, total_words, tokenizer = self.tokenize_data_inp_seq(train_data, result_path, each_run)
        padd_seq, max_len = self.pad_sequ(input_seq)
        xs, ys, labels = self.prep_seq_labels(padd_seq, total_words)

        self.train_model_five_runs_opt(total_words, max_len, xs, ys, result_path, test_data, model_number, each_run, logs_path)

    def get_available_cores(self, threshold=50, num_cores=16):
        usage_per_core = psutil.cpu_percent(interval=1, percpu=True)
        available = [i for i, usage in enumerate(usage_per_core) if usage < threshold]
        return available[:num_cores]

# Example usage
cl_ob = BiLSTMCybera()
sample = ("/mnt/siwuchuk/vscode/output_train", "/mnt/siwuchuk/vscode/models/bilstm/model/30/", "/mnt/siwuchuk/vscode/output_test", 30, "/mnt/siwuchuk/vscode/models/bilstm/logs/30")
cl_ob.consolidate_data_train_parallel(*sample)