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
    """Safely updates the embedding layer with new vocabulary size"""
    try:
        # Get original embedding configuration
        old_embedding = model.layers[0]
        old_weights = old_embedding.get_weights()[0]
        embedding_dim = old_weights.shape[1]
        
        # Calculate how many new tokens we need to add
        current_vocab_size = old_weights.shape[0]
        num_new_tokens = new_vocab_size - current_vocab_size
        
        if num_new_tokens <= 0:
            return model  # No update needed
        
        # Create new weights with proper initialization
        new_weights = np.vstack([
            old_weights,
            np.random.normal(
                loc=0.0,
                scale=0.01,
                size=(num_new_tokens, embedding_dim)
            )
        ])
        
        # Create new embedding layer
        new_embedding = Embedding(
            input_dim=new_vocab_size,
            output_dim=embedding_dim,
            weights=[new_weights],
            mask_zero=old_embedding.mask_zero,
            name='embedding'
        )
        
        # Rebuild model architecture
        if isinstance(model, Sequential):
            new_model = Sequential()
            new_model.add(new_embedding)
            for layer in model.layers[1:]:
                new_model.add(layer)
        else:
            input_layer = Input(shape=(None,), dtype='int32')
            x = new_embedding(input_layer)
            for layer in model.layers[1:]:
                x = layer(x)
            new_model = Model(inputs=input_layer, outputs=x)
        
        # Copy weights for other layers
        for i in range(1, len(new_model.layers)):
            if model.layers[i].get_weights():
                new_model.layers[i].set_weights(model.layers[i].get_weights())
        
        return new_model
    
    except Exception as e:
        print(f"Error updating embedding layer: {str(e)}")
        return model  # Return original model if update fails

def predict_masked_token_safely(masked_sequence, mask_pos, tokenz, model, maxlen):
    """Safely predicts a masked token using bidirectional context"""
    try:
        # Get left and right contexts
        tokens = masked_sequence.split()
        left_context = ' '.join(tokens[:mask_pos])
        right_context_reversed = ' '.join(reversed(tokens[mask_pos+1:]))
        
        # Get predictions from both directions
        left_pred, left_top = predict_token_score_upd_opt3(left_context, tokenz, model, maxlen)
        right_pred, right_top = predict_token_score_upd_opt3(right_context_reversed, tokenz, model, maxlen)
        
        # Combine predictions with weighting
        combined_scores = defaultdict(float)
        for token, score in left_top:
            combined_scores[token] += score * 0.6  # Higher weight for left context
        for token, score in right_top:
            combined_scores[token] += score * 0.4
        
        # Get top predictions
        if not combined_scores:
            return "UNK", [("UNK", 1.0)]  # Fallback
        
        predicted_token = max(combined_scores.items(), key=lambda x: x[1])[0]
        top_tokens = heapq.nlargest(10, combined_scores.items(), key=lambda x: x[1])
        
        return predicted_token, top_tokens
    
    except Exception as e:
        print(f"Error in masked token prediction: {str(e)}")
        return "UNK", [("UNK", 1.0)]

def predict_token_score_upd_opt3(context, tokenz, model, maxlen):
    """Robust token prediction with bounds checking"""
    try:
        # Get model's vocabulary capacity
        max_valid_index = model.layers[0].input_dim - 1
        
        # Tokenize input with bounds checking
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return "UNK", [("UNK", 1.0)]
        
        # Prepare base sequence
        base_sequence = [min(idx, max_valid_index) for idx in token_list[0][-maxlen + 1:]]
        
        # Get valid vocabulary
        vocab = []
        token_indices = []
        for token in tokenz.word_index:
            idx = tokenz.word_index[token]
            if idx <= max_valid_index:
                vocab.append(token)
                token_indices.append(idx)
        
        if not vocab:
            return "UNK", [("UNK", 1.0)]
        
        # Create input tensor
        input_seq = pad_sequences([base_sequence], maxlen=maxlen-1, padding='pre')
        input_tensor = tf.convert_to_tensor(input_seq)
        
        # Get predictions
        predictions = model(input_tensor, training=False)
        
        # Handle different output shapes
        if len(predictions.shape) == 2:
            logits = predictions[0]  # (batch_size, vocab_size)
        elif len(predictions.shape) == 3:
            logits = predictions[0, -1, :]  # (batch_size, seq_len, vocab_size)
        else:
            return "UNK", [("UNK", 1.0)]
        
        # Get probabilities for valid tokens
        valid_logits = tf.gather(logits, token_indices)
        probs = tf.nn.softmax(valid_logits).numpy()
        
        # Pair tokens with probabilities
        token_probs = list(zip(vocab, probs))
        top_tokens = heapq.nlargest(10, token_probs, key=lambda x: x[1])
        
        return top_tokens[0][0], top_tokens
    
    except Exception as e:
        print(f"Error in token prediction: {str(e)}")
        return "UNK", [("UNK", 1.0)]

def check_available_rank(list_tuples, true_word):
    """Check rank of true word in predictions"""
    if not list_tuples or not true_word:
        return -1
        
    true_word_clean = true_word.strip()
    for rank, (token, _) in enumerate(list_tuples, 1):
        if token.strip() == true_word_clean:
            return rank
    return -1

def find_resume_point(test_file_path, log_entry_count):
    """Find where to resume evaluation"""
    try:
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
            return None
    except Exception as e:
        print(f"Error finding resume point: {str(e)}")
        return None

# Run evaluation
evaluate_bilstm_masked_prediction(
    "/mnt/siwuchuk/thesis/another/kenlm/output_test/m10/scratch_test_set_10_6_1_proc_m.txt",
    47,
    "/mnt/siwuchuk/thesis/another/bilstm/models/10/main_bilstm_scratch_model_150embedtime1_main_sample_project10_6_1.keras",
    "/mnt/siwuchuk/thesis/another/bilstm/models/10/",
    10,
    1,
    "/mnt/siwuchuk/thesis/another/bilstm/logs/10_masked"
)