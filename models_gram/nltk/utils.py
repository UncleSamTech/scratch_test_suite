import os
from nltk import word_tokenize
import pickle
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
        print(f"Predicted next token: {scratch_predicted_next_token}")  # Debugging: Check predicted token
        
        # Get the top 10 tokens (sorted only once)
        top_10_tokens_scores = sorted(scratch_next_probaility_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 tokens: {top_10_tokens_scores}")  # Debugging: Check top 10 tokens
        
        return scratch_predicted_next_token, top_10_tokens_scores

#check_available_rank([("looksunderscoreswitchbackdropto",0.50),("backdrop",0.25),("looksunderscoreswitchbackdropto",0.15),("leftangliteralrightang",0.10)],"backdrop")

#gener_list_list(["eventunderscorewhenflagclicked", "eventunderscorewhenflagclicked looksunderscoreswitchbackdropto","eventunderscorewhenflagclicked looksunderscoreswitchbackdropto backdrop leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang"])
predict_next_scratch_token_upd_opt_small("/Users/samueliwuchukwu/desktop/analysis/models/nltk/nltk_10_2_1.pkl","eventunderscorewhenkeypressed")
