import os
from nltk import word_tokenize
import pickle
import kenlm
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

        print(f"predicted next token {predicted_next_token}, top 10 tokens {top_10_tokens_scores}")
        #returns predicted next token and list of top 10 tokens and scores
        return predicted_next_token,top_10_tokens_scores

#check_available_rank([("looksunderscoreswitchbackdropto",0.50),("backdrop",0.25),("looksunderscoreswitchbackdropto",0.15),("leftangliteralrightang",0.10)],"backdrop")

#gener_list_list(["eventunderscorewhenflagclicked", "eventunderscorewhenflagclicked looksunderscoreswitchbackdropto","eventunderscorewhenflagclicked looksunderscoreswitchbackdropto backdrop leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang"])
#predict_next_scratch_token_upd_opt_small("/Users/samueliwuchukwu/desktop/analysis/models/nltk/nltk_10_2_1.pkl","eventunderscorewhenkeypressed")
#predict_next_token_kenlm_upd("/Users/samueliwuchukwu/desktop/analysis/models/kenlm/kenln_10_2_1.arpa","eventunderscorewhenflagclicked controlunderscoreif condition","/Users/samueliwuchukwu/desktop/analysis/models/kenlm/kenln_10_2_1.vocab")
predict_next_token_kenlm_upd("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/arpa_files/10/kenln_10_2_1.arpa","eventunderscorewhenflagclicked controlunderscoreif condition","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/models/kenlm/vocab_files/10/kenln_10_2_1.vocab")