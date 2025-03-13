import os
from nltk import word_tokenize
import pickle
import kenlm
import tensorflow as tf
import heapq
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import time
#from nltk.lm import MLE,Vocabulary

def gener_list_list(data):
        if not data:
            return [[]]
        res = [list(word_tokenize(line.strip())) for line in data if line.strip()]
        print(res)
        return res

def check_available_rank(list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                print(rank)
                return rank
        
        return rank
def load_trained_model(model_name) :
        with open(model_name,"rb") as f:
            loaded_scratch_model = pickle.load(f)
            
        return loaded_scratch_model

def predict_next_scratch_token_upd_opt_small(model_name, context_data):
        loaded_model = load_trained_model(model_name)
        
        #print(f"Model loaded: {loaded_model}")  # Debugging: Check if model is loaded correctly
        
        context_tokens = context_data.split()  # Avoid repeated splits
        #print(f"Context tokens: {context_tokens}")  # Debugging: Check if context tokens are correct
        
        scratch_next_probaility_tokens = {
            token: loaded_model.score(token, context_tokens)
            for token in loaded_model.vocab
        }
        #print(f"Token probabilities: {scratch_next_probaility_tokens}")  # Debugging: Check token probabilities
        
        # Get the top predicted token
        scratch_predicted_next_token = max(scratch_next_probaility_tokens, key=scratch_next_probaility_tokens.get)
        # print(f"Predicted next token: {scratch_predicted_next_token}")  # Debugging: Check predicted token
        
        # Get the top 10 tokens (sorted only once)
        top_10_tokens_scores = sorted(scratch_next_probaility_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        # print(f"Top 10 tokens: {top_10_tokens_scores}")  # Debugging: Check top 10 tokens
        
        return scratch_predicted_next_token, top_10_tokens_scores

def predict_next_token_kenlm_upd(model, context,vocab_name):
        mod = kenlm.Model(model)
        
        next_token_probabilities = {}
        
        
        with open(vocab_name, "r", encoding="utf8") as vocab_f:
                vocabulary = vocab_f.readlines()
                for candidate_word in vocabulary:
                    candidate_word = candidate_word.strip()
                    context_with_candidate = context + " " + candidate_word
                    next_token_probabilities[candidate_word] = mod.score(context_with_candidate)
                    

        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)
        
        top_10_tokens_scores = sorted(next_token_probabilities.items(), key=lambda item: item[1], reverse=True)[:10]

        # print(f"predicted next token {predicted_next_token}, top 10 tokens {top_10_tokens_scores}")
        #returns predicted next token and list of top 10 tokens and scores
        return predicted_next_token,top_10_tokens_scores

def predict_token_score_upd_opt(context, model_file, maxlen,tokenizer_file):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.
        Optimized to reduce redundant computations and improve efficiency.
        """
        start= time.time()
        # Load tokenized data once
        with open(tokenizer_file, "rb") as tk:
            tokenz = pickle.load(tk)
        
        loaded_model = load_model(model_file, compile=False)

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
        predictions = loaded_model.predict(padded_sequences, verbose=0)

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
        #print(f"true token {predicted_next_token}\n top 10 tokens : {top_10_tokens_scores}")
        end = time.time() - start
        print(f"time : {end}")
        print(f"pred token {predicted_next_token}, top 10 tokens : {top_10_tokens_scores}")
        return predicted_next_token, top_10_tokens_scores





def predict_token_score_upd_opt2(context, model_file, maxlen,tokenizer_file):
    start= time.time()
    """
    Predicts the next token based on the given context and scores each token in the vocabulary.
    Optimized to reduce redundant computations and improve efficiency.
    """
    
    # Load tokenized data once
    with open(tokenizer_file, "rb") as tk:
        tokenz = pickle.load(tk)
        
    loaded_model = load_model(model_file, compile=False)
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
    predictions = loaded_model(padded_sequences, training=False)  # Use model.call() for raw logits

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

    end = time.time() - start
    print(f"time : {end}")
    print(f"pred token {predicted_next_token}, top 10 tokens : {top_10_tokens_scores}")
    return predicted_next_token, top_10_tokens_scores


#check_available_rank([("looksunderscoreswitchbackdropto",0.50),("backdrop",0.25),("looksunderscoreswitchbackdropto",0.15),("leftangliteralrightang",0.10)],"backdrop")

#gener_list_list(["eventunderscorewhenflagclicked", "eventunderscorewhenflagclicked looksunderscoreswitchbackdropto","eventunderscorewhenflagclicked looksunderscoreswitchbackdropto backdrop leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang"])
#predict_next_scratch_token_upd_opt_small("/Users/samueliwuchukwu/desktop/analysis/models/nltk/nltk_10_2_1.pkl","eventunderscorewhenkeypressed")
#predict_next_token_kenlm_upd("/Users/samueliwuchukwu/desktop/analysis/models/kenlm/kenln_10_2_1.arpa","eventunderscorewhenflagclicked controlunderscoreif condition","/Users/samueliwuchukwu/desktop/analysis/models/kenlm/kenln_10_2_1.vocab")
#predict_next_token_kenlm_upd("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/10/kenln_10_2_1.arpa","eventunderscorewhenflagclicked controlunderscoreif condition","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/10/kenln_10_2_1.vocab")

predict_token_score_upd_opt("eventunderscorewhenflagclicked controlunderscoreifunderscoreelse","/Users/samueliwuchukwu/desktop/analysis/models/bilstm/main_bilstm_scratch_model_150embedtime1_main_sample_project30_6_1.keras",47,"/Users/samueliwuchukwu/desktop/analysis/models/bilstm/tokenized_file_50embedtime1_1.pickle")
predict_token_score_upd_opt2("eventunderscorewhenflagclicked controlunderscoreifunderscoreelse","/Users/samueliwuchukwu/desktop/analysis/models/bilstm/main_bilstm_scratch_model_150embedtime1_main_sample_project30_6_1.keras",47,"/Users/samueliwuchukwu/desktop/analysis/models/bilstm/tokenized_file_50embedtime1_1.pickle")