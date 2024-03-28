import os
import kenlm
import sys
import nltk
import numpy as np
import subprocess
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_curve,f1_score

class kenlm_train:

    def __init__(self):
        self.result = []
        self.tokenized_data = ""

    def tokenize_kenlm(self,train_data):
        for line in train_data:
            for sentence in nltk.sent_tokenize(line):
                token_sentlist = nltk.word_tokenize(sentence)
                self.tokenized_data += ''.join(token_sentlist).lower()
        return self.tokenized_data

    
    def access_train_data_kenlm(self,file_path,cwd):
        if os.path.isfile(file_path):
            with open(file_path,"r") as each_sentence:
                each_line = each_sentence.readlines()
                val = self.tokenize_kenlm(each_line)
                print(val)
                #module_train = subprocess.run(['/mnt/c/Users/USER/Documents/model_train/online/kenlm/build/bin/lmplz -o 3 > kenlm.arpa'],stdin=val,stdout=subprocess.PIPE, cwd=cwd, shell=True)


    def test_kenlm(self,arpa_file):
        model = kenlm.Model(arpa_file)
        print(model.score("event_whenflagclicked",bos=True,eos=True))
        return model
    
    def replace_non_vs_string_with_tokens(self,string_val):
        if isinstance(string_val,str) and len(string_val) > 0:
            val2 = string_val.split()
            print("see tokens" , val2)
            new_list = ['<S>' if word not in self.valid_opcodes and word not in self.valid_other_field_codes  else word for word in val2  ]
            print("replaced tokens" , new_list)
            return " ".join(new_list)
        else:
            return ""

    
    def scratch_evaluate_model_kenlm(self,test_data,model_name):

        y_true = []
        i=0
        y_pred = []
        model = kenlm.Model(model_name)

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            
            for line in lines:
                #line = self.replace_non_vs_string_with_tokens(line)
                line = line.strip()
                sentence_tokens = line.split()
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]
            
                predicted_next_word = self.predict_next_token_kenlm(model,context)
                
                
                i+=1
                if i%500 == 0:
                    print(f"progress {i} true next word {true_next_word} predicted next word {predicted_next_word}")
            
                y_true.append(true_next_word)
                y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        f1score = f1_score(y_true,y_pred,average="weighted")
        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_acc_prec_rec_f1.txt","a") as frp:
            frp.write(f" order 5 accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
            frp.write("\n")
        return accuracy,precision,recall,f1score


    def create_vocab(self,arpa_file,vocab_file):
        with open(arpa_file,"r",encoding="utf-8") as fr:
            lines = fr.readlines()
            one_grams_seen = False
            i = 0
            for line in lines:
                
                line=line.strip()
                
                if "\\1-grams" in line:
                    one_grams_seen = True
                    continue
                if one_grams_seen:
                    with open(vocab_file,"a") as vf:
                        token = line.split("\t")[1]
                        
                        vf.write(token+"\n") 
                    
                   
                


    def predict_next_token_kenlm(self,model, context):
    #context_tokens = context.split(" ")
        next_token_probabilities = {}

        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/vocabs_folder/kenlm_sb3_order2.vocab", "r", encoding="utf8") as vocab_f:
            vocabulary = vocab_f.readlines()
            for candidate_word in vocabulary:
                candidate_word = candidate_word.strip()
                context_with_candidate = context + " " + candidate_word
                next_token_probabilities[candidate_word] = model.score(context_with_candidate)

        predicted_next_token = max(next_token_probabilities, key=next_token_probabilities.get)
        return predicted_next_token
kn = kenlm_train()

#kn.create_vocab("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order10.arpa","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/vocabs_folder/kenlm_sb3_order2.vocab")
#print(kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa"))
#model_evaluated = kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa")
val = kn.scratch_evaluate_model_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order5.arpa")
print(val)
#print(kn.access_train_data_kenlm("scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/online/kenlm/build")) 

#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 2  --discount_fallback < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order2.arpa       