import pandas as pd
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time
import seaborn as sns
import re
import psutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable GPU and configure TensorFlow for CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.threading.set_intra_op_parallelism_threads(1)  # Limit intra-op parallelism
tf.config.threading.set_inter_op_parallelism_threads(1)  # Limit inter-op parallelism
K.clear_session()  # Clear any existing sessions

class bilstm_cybera:
    def consolidate_data_train_parallel(self, train_path, result_path, test_path, model_number, logs_path):
        """
        Spawns a separate process for each run and assigns 1 CPU core per run.
        Spreads 5 runs across all available cores.
        """
        skipped_run = [3, 5]
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for each_run in range(2, 6):  # 5 runs
                if each_run in skipped_run:
                    continue

                # Submit tasks to the executor
                future = executor.submit(
                    self.run_consolidate_train_run,
                    train_path, result_path, test_path, model_number, logs_path, each_run, [each_run % multiprocessing.cpu_count()]
                )
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()

    def train_model_five_runs_opt(self, total_words, max_seq, xs, ys, result_path, test_data, proj_number, runs, logs_path):
        """
        Trains the BiLSTM model with optimized data pipeline and callbacks.
        """
        # Create a tf.data.Dataset for efficient batching and prefetching
        dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
        dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

        # Define the model
        model = Sequential([
            Input(shape=(max_seq - 1,)),  # Explicitly define the input shape
            Embedding(total_words, 50),  # Reduced embedding dimension
            Bidirectional(LSTM(100)),  # Reduced LSTM units
            Dense(total_words, activation='softmax')
        ])

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

        # Define callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Train the model
        history = model.fit(dataset, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

        # Save the history and model
        with open(f"{result_path}main_historyrec_150embedtime_6_{runs}.pickle", "wb") as hs:
            pickle.dump(history.history, hs)

        file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_sample_project{proj_number}_6_{runs}.keras"
        if os.path.exists(file_name):
            os.remove(file_name)
        model.save(file_name)

    def tokenize_data_inp_seq(self, file_name, result_path, run, chunk_size=100000):
        """
        Tokenizes the input data in chunks to reduce memory usage.
        """
        self.tokenizer = Tokenizer(oov_token='<oov>')
        self.encompass = []

        # Use a generator to read the file in chunks
        def read_chunks(file_name, chunk_size):
            with open(file_name, "r", encoding="utf-8") as rf:
                while True:
                    lines = rf.readlines(chunk_size)
                    if not lines:
                        break
                    yield lines

        # Fit the tokenizer and process data in chunks
        for lines in read_chunks(file_name, chunk_size):
            self.tokenizer.fit_on_texts(lines)
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
        logger.info(f"Total words (vocabulary size): {self.total_words}")

        return self.encompass, self.total_words, self.tokenizer

    def pad_sequ(self, input_seq):
        """
        Pads the input sequences to the same length.
        """
        max_seq_len = max([len(x) for x in input_seq])
        padded_in_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding='pre'))
        return padded_in_seq, max_seq_len

    def prep_seq_labels(self, padded_seq, total_words):
        """
        Prepares input sequences and labels for training.
        """
        xs, labels = padded_seq[:, :-1], padded_seq[:, -1]

        max_label_index = np.max(labels)
        if max_label_index >= total_words:
            logger.warning(f"Adjusting total_words from {total_words} to {max_label_index + 1} based on labels.")
            total_words = max_label_index + 1

        if np.any(labels >= total_words):
            raise ValueError(f"Labels contain indices >= total_words: {np.max(labels)} >= {total_words}")

        ys = to_categorical(labels, num_classes=total_words)
        return xs, ys, labels

    def run_consolidate_train_run(self, train_path, result_path, test_path, model_number, logs_path, each_run, cores):
        """
        Sets CPU affinity for this process and performs one run of training.
        """
        # Set CPU affinity
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cores)
        logger.info(f"[PID {os.getpid()}] Running run {each_run} on cores {cores}")

        # Construct file paths using pathlib
        train_data = Path(train_path) / f"scratch_train_set_{model_number}_6_{each_run}_proc.txt"
        test_data = Path(test_path) / f"scratch_test_set_{model_number}_6_{each_run}_proc.txt"

        # Run the sequence of operations
        input_seq, total_words, tokenizer = self.tokenize_data_inp_seq(train_data, result_path, each_run)
        padd_seq, max_len = self.pad_sequ(input_seq)
        xs, ys, labels = self.prep_seq_labels(padd_seq, total_words)
        logger.info(f"Maximum length for run {each_run}: {max_len}")

        self.train_model_five_runs_opt(total_words, max_len, xs, ys, result_path, test_data, model_number, each_run, logs_path)

# Example usage
if __name__ == "__main__":
    cl_ob = bilstm_cybera()
    sample = (
        "/mnt/siwuchuk/vscode/output_train",
        "/mnt/siwuchuk/vscode/models/bilstm/model/30/",
        "/mnt/siwuchuk/vscode/output_test",
        30,
        "/mnt/siwuchuk/vscode/models/bilstm/logs/30"
    )
    cl_ob.consolidate_data_train_parallel(*sample)