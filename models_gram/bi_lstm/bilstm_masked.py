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
    """
    More robust embedding layer update with comprehensive error handling and validation
    """
    try:
        # Verify the model has layers
        if not model.layers or len(model.layers) == 0:
            print("Error: Model has no layers")
            return model
            
        # Get original embedding configuration
        old_embedding = model.layers[0]
        
        # Ensure first layer is an embedding layer
        if not isinstance(old_embedding, Embedding):
            print("Error: First layer is not an Embedding layer")
            return model
            
        # Get weights with validation
        if not old_embedding.get_weights():
            print("Error: Embedding layer has no weights")
            return model
            
        old_weights = old_embedding.get_weights()[0]
        embedding_dim = old_weights.shape[1]
        
        # Check if vocabulary update is needed
        current_vocab_size = old_weights.shape[0]
        if new_vocab_size <= current_vocab_size:
            print(f"No embedding update needed: current size {current_vocab_size} >= new size {new_vocab_size}")
            return model
            
        # Create new weights - ensure we don't exceed max vocabulary or cause memory issues
        num_new_tokens = min(
            new_vocab_size - current_vocab_size,
            10000  # Safety limit
        )
        
        print(f"Extending embedding layer: {current_vocab_size} → {current_vocab_size + num_new_tokens}")
        
        # Generate new weights with proper initialization
        try:
            new_weights = np.vstack([
                old_weights,
                np.random.normal(
                    loc=0.0,
                    scale=0.01,  # Small standard deviation for stable initialization
                    size=(num_new_tokens, embedding_dim)
                )
            ])
        except Exception as e:
            print(f"Error creating weights array: {str(e)}")
            return model
        
        # Rebuild model with memory efficiency in mind
        try:
            # Create new embedding layer
            new_embedding = Embedding(
                input_dim=current_vocab_size + num_new_tokens,
                output_dim=embedding_dim,
                weights=[new_weights],
                mask_zero=old_embedding.mask_zero,
                name='embedding'
            )
            
            # Rebuild model based on its type
            if isinstance(model, Sequential):
                # For Sequential models
                new_model = Sequential()
                new_model.add(new_embedding)
                
                # Add remaining layers
                for layer in model.layers[1:]:
                    new_model.add(layer)
            else:
                # For Functional API models
                input_layer = Input(shape=(None,), dtype='int32')
                x = new_embedding(input_layer)
                
                # Reconstruct layer chain
                for layer in model.layers[1:]:
                    x = layer(x)
                
                new_model = Model(inputs=input_layer, outputs=x)
            
            # Copy weights for other layers
            for i in range(1, len(new_model.layers)):
                if i < len(model.layers) and model.layers[i].get_weights():
                    new_model.layers[i].set_weights(model.layers[i].get_weights())
            
            # Recompile if needed
            if hasattr(model, 'optimizer') and model.optimizer:
                try:
                    new_model.compile(
                        optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=model.metrics
                    )
                except Exception as e:
                    print(f"Warning: Could not compile model: {str(e)}")
                    # Continue without compilation as the model can still be used for inference
            
            # Clear backend session to free memory
            K.clear_session()
            
            return new_model
            
        except Exception as e:
            print(f"Error rebuilding model: {str(e)}")
            return model
            
    except Exception as e:
        print(f"Error in embedding layer update: {str(e)}")
        return model  # Return original model on error
    

# Fix 1: The predict_masked_token_safely function with empty dictionary handling
def predict_masked_token_safely(masked_sequence, mask_pos, tokenz, model, maxlen):
    """
    Safely predicts a masked token using bidirectional context.
    Handles empty prediction results gracefully.
    """
    # Get max valid index from model
    max_index = model.layers[0].input_dim
    
    # Tokenize with bounds checking
    tokens = masked_sequence.split()
    token_ids = [min(tokenz.word_index.get(token, len(tokenz.word_index)), max_index-1) for token in tokens]
    
    # Get left and right contexts
    left_context = ' '.join(tokens[:mask_pos])
    right_context_reversed = ' '.join(reversed(tokens[mask_pos+1:]))
    
    # Get predictions
    left_pred, left_top = predict_token_score_upd_opt3(left_context, tokenz, model, maxlen)
    right_pred, right_top = predict_token_score_upd_opt3(right_context_reversed, tokenz, model, maxlen)
    
    # Combine predictions
    combined_scores = defaultdict(float)
    for token, score in left_top:
        combined_scores[token] += score * 0.6
    for token, score in right_top:
        combined_scores[token] += score * 0.4
    
    # Handle empty combined_scores
    if not combined_scores:
        # Fallback strategy: try left prediction, then right, then default
        if isinstance(left_pred, str) and left_pred != -1:
            predicted_token = left_pred
            top_tokens = left_top if left_top else [(left_pred, 1.0)]
        elif isinstance(right_pred, str) and right_pred != -1:
            predicted_token = right_pred
            top_tokens = right_top if right_top else [(right_pred, 1.0)]
        else:
            # Last resort fallback
            predicted_token = "UNK"
            top_tokens = [("UNK", 1.0)]
    else:
        # Normal case - we have scores
        predicted_token = max(combined_scores.items(), key=lambda x: x[1])[0]
        top_tokens = heapq.nlargest(10, combined_scores.items(), key=lambda x: x[1])
    
    return predicted_token, top_tokens

# Fix 2: Improved predict_token_score_upd_opt3 function with robust error handling
def predict_token_score_upd_opt3(context, tokenz, model, maxlen):
    """
    Fully robust prediction function with proper tensor handling and error recovery
    """
    try:
        # Handle empty context
        if not context or context.strip() == "":
            return "UNK", [("UNK", 1.0)]
            
        # Get model's vocabulary capacity
        max_valid_index = model.layers[0].input_dim - 1
        
        # Tokenize input with bounds checking
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return "UNK", [("UNK", 1.0)]
        
        # Prepare base sequence
        base_sequence = [min(idx, max_valid_index) for idx in token_list[0][-maxlen + 1:]]
        if not base_sequence:
            return "UNK", [("UNK", 1.0)]
        
        # Get valid vocabulary
        try:
            vocab = [t for t in tokenz.word_index if tokenz.word_index[t] <= max_valid_index]
            if not vocab:
                return "UNK", [("UNK", 1.0)]
        except Exception as e:
            print(f"Vocabulary error: {str(e)}")
            return "UNK", [("UNK", 1.0)]
        
        # Create proper input tensor
        input_seq = pad_sequences([base_sequence], maxlen=maxlen-1, padding='pre')
        input_tensor = tf.convert_to_tensor(input_seq)
        
        # Get model's prediction with error handling
        try:
            predictions = model(input_tensor, training=False)
            
            # Handle different model output formats
            if len(predictions.shape) == 2:
                # Sequential model output (batch_size, vocab_size)
                logits = predictions[0]  # Get first (and only) batch item
            elif len(predictions.shape) == 3:
                # Many-to-many output (batch_size, seq_len, vocab_size)
                logits = predictions[0, -1, :]  # Get last position
            else:
                print(f"Unexpected prediction shape: {predictions.shape}")
                return "UNK", [("UNK", 1.0)]
        except tf.errors.ResourceExhaustedError:
            print("Resource exhausted during prediction")
            return "UNK", [("UNK", 1.0)]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "UNK", [("UNK", 1.0)]
        
        # Filter logits for valid tokens only
        try:
            token_indices = [tokenz.word_index[t] for t in vocab]
            filtered_logits = tf.gather(logits, token_indices)
            
            # Convert to probabilities
            probs = tf.nn.softmax(filtered_logits).numpy()
            
            # Pair tokens with their probabilities
            token_probs = list(zip(vocab, probs))
            
            # Get top predictions
            top_tokens = heapq.nlargest(min(10, len(token_probs)), token_probs, key=lambda x: x[1])
            if not top_tokens:
                return "UNK", [("UNK", 1.0)]
                
            predicted_token = top_tokens[0][0]
            return predicted_token, top_tokens
        except Exception as e:
            print(f"Token scoring error: {str(e)}")
            return "UNK", [("UNK", 1.0)]
    
    except Exception as e:
        print(f"General prediction function error: {str(e)}")
        return "UNK", [("UNK", 1.0)]

# Fix 3: The check_available_rank function to better handle edge cases
def check_available_rank(list_tuples, true_word):
    """
    Find the rank of the true word in the prediction list.
    Returns -1 if not found.
    """
    if not list_tuples or not true_word:
        return -1
        
    rank = -1
    true_word_clean = true_word.strip()
    
    for ind, val in enumerate(list_tuples):
        # Handle cases where val might not be a tuple
        if not isinstance(val, tuple) or len(val) < 1:
            continue
            
        token = val[0]
        if not isinstance(token, str):
            continue
            
        if true_word_clean == token.strip():
            rank = ind + 1
            return rank
    return rank

# Fix 4: Update find_resume_point to handle file not found
def find_resume_point(test_file_path, log_entry_count):
    """Find the line and token position in the test file to resume evaluation."""
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
            print(f"Total lines processed in test file: {line_num+1}")
        return None
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file_path}")
        return None
    except Exception as e:
        print(f"Error finding resume point: {str(e)}")
        return None

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