import os
import pickle
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
import nltk
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score

class scratch_train_mle:

    def __init__(self):
        self.scratch_model = None
        self.ngram_model = None
        self.loaded_scratch_model = None

    def train_mle(self,train_data,n,trained_model_data):

        with open(train_data,"r",encoding="utf-8") as f:
            lines = f.readlines()
            #sublines = lines[:10000]
            tokenized_scratch_data = [list(word_tokenize(sent.strip())) for sent in lines]
            train_data,padded_sents = padded_everygram_pipeline(n,tokenized_scratch_data)
        
        try:
            self.scratch_model = MLE(n)
            self.scratch_model.fit(train_data,padded_sents)

            with open(f'{trained_model_data}_{n}.pkl',"wb") as fd:
                pickle.dump(self.scratch_model,fd)
        except Exception as es:
            print("error as a result of ", es)

           
    def load_trained_model(self,model_name) :
        with open(model_name,"rb") as f:
            self.loaded_scratch_model = pickle.load(f)
            #print(type(self.loaded_scratch_model))
            #print(self.loaded_scratch_model.vocab)
            #print(self.loaded_scratch_model.counts("event_whenflagclicked"))
            #print(self.loaded_scratch_model.score("event_whenflagclicked"))
            #print(self.loaded_scratch_model.vocab.lookup("event_whenflagclicked"))
        return self.loaded_scratch_model
    
   

    def predict_next_scratch_token(self,model_name,context_data):
        loaded_model = self.load_trained_model(model_name)
        scratch_next_probaility_tokens = {}

        for prospect_token in loaded_model.vocab:
            #print("see token" , prospect_token)
            scratch_next_probaility_tokens[prospect_token] = loaded_model.score(prospect_token,context_data.split(" "))
        
        scratch_predicted_next_token = max(scratch_next_probaility_tokens,key=scratch_next_probaility_tokens.get)
        return scratch_predicted_next_token
    
    def scratch_evaluate_model_nltk(self,test_data,model_name):

        y_true = []
        i=0
        y_pred = []

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            lines_lenght = len(lines)
            print("lenght",lines_lenght)
            offset_lenght = lines_lenght - 50
            new_lines = lines[:offset_lenght]
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]
            
                predicted_next_word = self.predict_next_scratch_token(model_name,context)
                with open("seelogs.txt","a") as fp:
                    fp.write(f"for context {context} next token {predicted_next_word}")
                    fp.write("\n")
                
                i+=1
                if i%500 == 0:
                    print("see it",i)
            
                y_true.append(true_next_word)
                y_pred.append(predicted_next_word)


        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        print(f"accuracy {accuracy} precisions {precision} recall {recall}")
        return accuracy,precision,recall
    
tr_scr = scratch_train_mle()
#tr_scr.train_mle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90.txt",4,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2")
tr_scr.load_trained_model("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_4.pkl")
#tr_scr.scratch_evaluate_model_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_3.pkl") 