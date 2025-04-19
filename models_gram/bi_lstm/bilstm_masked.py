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
from tensorflow.keras.models import Sequential, Model
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
from collections import defaultdict


def evaluate_bilstm_masked_prediction(test_data, maxlen, model, result_path, proj_number, run, logs_path, mask_prob=0.15):
    """Optimized BiLSTM evaluation with masked token prediction only."""
    try:
        # Load resources once with memory optimization
        loaded_model = load_model(model, compile=False)
        
        # Load tokenizer
        tokenized_file_path = f"{result_path}tokenized_file_50embedtime1_{run}.pickle"
        with open(tokenized_file_path, "rb") as tk:
            tokenz = pickle.load(tk)
        
        # Ensure [MASK] token exists
        if 'lbracmaskrbrac' not in tokenz.word_index:
            tokenz.word_index['lbracmaskrbrac'] = len(tokenz.word_index) + 1
        
        # Update model embedding layer
        new_vocab_size = len(tokenz.word_index) + 1
        loaded_model = update_embedding_layer_safely(loaded_model, new_vocab_size)
        
        # Log file setup
        investig_path = f"{logs_path}/bilstm_masked_{proj_number}_6_{run}_logs.txt"
        
        # Initialize log file
        if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
            with open(investig_path, "w") as log_file:
                log_file.write("masked_sequence,masked_position,true_token,predicted_token,rank,correct\n")
            log_entry_count = 0
        else:
            with open(investig_path, "r") as f:
                log_entry_count = sum(1 for _ in f) - 1

        # Resume logic
        resume_line, resume_token = 0, 1
        if log_entry_count > 0:
            resume_info = find_resume_point(test_data, log_entry_count)
            if resume_info is None:
                print("Evaluation already completed")
                return
            resume_line, resume_token = resume_info
            print(f"Resuming from line {resume_line+1}, token {resume_token+1}")

        # Process file
        with open(test_data, "r", encoding="utf-8") as f, \
            open(investig_path, "a") as log_file:
            
            # Skip to resume line
            for _ in range(resume_line):
                next(f)
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                tokens = line.split()
                if len(tokens) < 3:  # Need at least 1 token before and after mask
                    continue
                
                # Process tokens with masking
                for idx in range(resume_token, len(tokens)-1):  # Skip first/last tokens
                    # Only process with mask_prob probability
                    if random.random() >= mask_prob:
                        continue
                        
                    # Create masked sequence
                    true_word = tokens[idx]
                    masked_tokens = tokens.copy()
                    masked_tokens[idx] = 'lbracmaskrbrac'
                    masked_sequence = ' '.join(masked_tokens)
                    
                    # Get bidirectional prediction
                    pred, top_tokens = predict_masked_token_safely(
                        masked_sequence, idx, tokenz, loaded_model, maxlen
                    )
                    rank = check_available_rank(top_tokens, true_word)
                    
                    log_file.write(
                        f"{masked_sequence},{idx},{true_word},{pred},{rank},{int(true_word == pred)}\n"
                    )
                    
                    # Memory cleanup
                    if idx % 100 == 0:
                        gc.collect()
                
                # Reset resume token after first line
                resume_token = 1
    
    finally:
        # Cleanup
        if 'loaded_model' in locals():
            del loaded_model
        gc.collect()

def update_embedding_layer_safely(model, new_vocab_size):
    """Safely updates the embedding layer with new vocabulary size by rebuilding model"""
    # Get original model configuration
    old_embedding = model.layers[0]
    embedding_dim = old_embedding.output_dim
    old_weights = old_embedding.get_weights()[0]
    
    # Create new embedding weights
    new_weights = np.vstack([
        old_weights,
        np.random.normal(
            size=(new_vocab_size - old_weights.shape[0], 
            scale=0.01,
            loc=0.0)
        )
    ])
    
    # Rebuild model architecture
    input_layer = Input(shape=(None,), dtype='int32', name='input_layer')
    new_embedding = Embedding(
        input_dim=new_vocab_size,
        output_dim=embedding_dim,
        weights=[new_weights],
        mask_zero=old_embedding.mask_zero,
        name='embedding'
    )(input_layer)
    
    # Reconnect all subsequent layers
    prev_layer = new_embedding
    for layer in model.layers[1:]:
        prev_layer = layer(prev_layer)
    
    # Create and compile new model
    new_model = Model(inputs=input_layer, outputs=prev_layer)
    if model.optimizer:
        new_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics
        )
    
    return new_model

def check_available_rank(list_tuples, true_word):
    rank = -1
    for ind, val in enumerate(list_tuples):
        if true_word.strip() == val[0].strip():
            rank = ind + 1
            return rank
    return rank

def predict_masked_token_bidirectional(masked_sequence, mask_pos, tokenz, model, maxlen):
    """
    Predicts masked token using bidirectional context approximation.
    Returns:
        predicted_token: The top predicted token
        top_tokens: List of (token, score) tuples
    """
    tokens = masked_sequence.split()
    
    # Left context prediction
    left_context = ' '.join(tokens[:mask_pos])
    left_pred, left_top = predict_token_score_upd_opt2(left_context, tokenz, model, maxlen)
    
    # Right context prediction (reverse sequence)
    right_context = ' '.join(reversed(tokens[mask_pos+1:]))
    right_pred, right_top = predict_token_score_upd_opt2(right_context, tokenz, model, maxlen)
    
    # Combine predictions (average scores)
    combined_scores = defaultdict(float)
    for token, score in left_top:
        combined_scores[token] += score * 0.5
    for token, score in right_top:
        combined_scores[token] += score * 0.5
        
    predicted_token = max(combined_scores.items(), key=lambda x: x[1])[0]
    top_tokens = heapq.nlargest(10, combined_scores.items(), key=lambda x: x[1])
    
    return predicted_token, top_tokens

def find_resume_point(test_file_path, log_entry_count):
    """Find the line and token position in the test file to resume evaluation."""
    with open(test_file_path, 'r') as test_file:
        current_log_entries = 0
        for line_num, line in enumerate(test_file):
            tokens = line.strip().split()
            if len(tokens) >= 2:
                tokens_after_first = len(tokens) - 1
                if current_log_entries + tokens_after_first >= log_entry_count:
                    token_pos = log_entry_count - current_log_entries
                    return line_num, token_pos
                current_log_entries += tokens_after_first
        print(f"Total lines in test file: {current_log_entries}")
    return None

def predict_token_score_upd_opt2(context, tokenz, model, maxlen):
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
    predictions = model(padded_sequences, training=False)

    # Extract probabilities for each token
    max_prob_tokens = {
        token: predictions[i][token_index].numpy()
        for i, (token, token_index) in enumerate(zip(vocab, token_indices))
    }

    # Find the predicted next token
    predicted_next_token = max(max_prob_tokens, key=max_prob_tokens.get)

    # Get top-10 tokens
    top_10_tokens_scores = heapq.nlargest(10, max_prob_tokens.items(), key=lambda x: x[1])

    return predicted_next_token, top_10_tokens_scores

def safe_tokenize(text, tokenizer):
    """Converts text to token IDs, handling out-of-vocabulary tokens"""
    tokens = text.split()
    token_ids = []
    for token in tokens:
        idx = tokenizer.word_index.get(token, len(tokenizer.word_index))
        if idx >= len(tokenizer.word_index):
            idx = len(tokenizer.word_index) - 1
        token_ids.append(idx)
    return token_ids

def predict_masked_token_safely(masked_sequence, mask_pos, tokenz, model, maxlen):
    # Tokenize safely
    token_ids = safe_tokenize(masked_sequence, tokenz)
    
    # Get left and right contexts
    left_context_ids = token_ids[:mask_pos]
    right_context_ids = token_ids[mask_pos+1:]
    
    # Convert back to text
    left_context = ' '.join([list(tokenz.word_index.keys())[i] for i in left_context_ids])
    right_context_reversed = ' '.join([list(tokenz.word_index.keys())[i] for i in reversed(right_context_ids)])
    
    # Get predictions
    left_pred, left_top = predict_token_score_upd_opt2(left_context, tokenz, model, maxlen)
    right_pred, right_top = predict_token_score_upd_opt2(right_context_reversed, tokenz, model, maxlen)
    
    # Combine predictions
    combined_scores = defaultdict(float)
    for token, score in left_top:
        combined_scores[token] += score * 0.5
    for token, score in right_top:
        combined_scores[token] += score * 0.5
    
    predicted_token = max(combined_scores.items(), key=lambda x: x[1])[0]
    top_tokens = heapq.nlargest(10, combined_scores.items(), key=lambda x: x[1])
    
    return predicted_token, top_tokens


# Run the evaluation
evaluate_bilstm_masked_prediction(
    "/mnt/siwuchuk/thesis/another/kenlm/output_test/m10/scratch_test_set_10_6_1_proc_m.txt",
    47,
    "/mnt/siwuchuk/thesis/another/bilstm/models/10/main_bilstm_scratch_model_150embedtime1_main_sample_project10_6_1.keras",
    "/mnt/siwuchuk/thesis/another/bilstm/models/10/",
    10,
    1,
    "/mnt/siwuchuk/thesis/another/bilstm/logs/10_masked"
)