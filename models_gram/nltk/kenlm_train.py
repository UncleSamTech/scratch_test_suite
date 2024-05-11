import os
#import kenlm
import sys
import nltk
import numpy as np
import subprocess
import random
import scipy.stats as stats

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
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
            frp.write(f" order 7 accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
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
    
    def plot_precision_recall_curve(self,plot_name):

        Accuracy = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        Precision = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737]
        Recall = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        F1 = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169]
        Ngrams = [2,3,4,5,6]

        Accuracy2 = [0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004]
        Precision2 = [0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7419880416193364]
        Recall2 = [0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004]
        F1_2 = [0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5352446315786005]
        Ngrams2 = [7,8,9,10,11]
        
        Accuracy3 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision3 = [0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall3 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_3 = [0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815]
        Ngrams3 = [12,13,14,15,16]

        Accuracy4 = [0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision4 = [0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall4 = [0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_4 = [0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815]
        Ngrams4 = [17,18,19,20]

        Accuracy_all = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,
                        0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004,
                        0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,
                        0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        Precision_all = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,
                         0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7419880416193364,
                         0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916,
                         0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916]
        Recall_all = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,
                      0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5743961352657004,
                      0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,
                      0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609]
        F1_all  = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,
                   0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5352446315786005,
                   0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815,
                   0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815
                   ]
        Ngrams_all = list(range(2,21))
        #[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        #fig = plt.figure()
        #axes = fig.add_axes([0,0,1,1])
        #axes.plot(Ngrams_all,Accuracy_all,label = "Accuracy")
        #axes.plot(Ngrams_all,Precision_all,label = "Precision")
        #axes.plot(Ngrams_all,Recall_all,label="Recall")
        #axes.plot(Ngrams_all,F1_all,label="F1")
        #axes.legend(loc ="center right")

        
        plt.plot(Ngrams_all, Precision_all, label = "Precision")
        plt.plot(Ngrams_all, Recall_all, label = "Recall")
        plt.plot(Ngrams_all, F1_all, label = "F1")
        plt.plot(Ngrams_all, Accuracy_all, label = "Accuracy")
        
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Kenlm_Model Scores vs N-Gram Orders for replaced tokens')
        plt.legend()
        #plt.xlim(0,21)
        #plt.ylim(0,0.79)
        #plt.show()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}.pdf')

    def paired_t_test(self,nltk_2_10,nltk_11_19):
        if isinstance(nltk_2_10,list) and len(nltk_2_10) > 0 and isinstance(nltk_11_19,list) and len(nltk_11_19) > 0:
            test_val = stats.ttest_rel(nltk_2_10,nltk_11_19)
            #print(test_val)
            return test_val
        
    def wilcon_t_test(self,group1,group2):
        return stats.wilcoxon(group1,group2)
kn = kenlm_train()


#kn.create_vocab("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order10.arpa","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/vocabs_folder/kenlm_sb3_order2.vocab")
#print(kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa"))
#model_evaluated = kn.test_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas_upd/kenlm_order2_model.arpa")
#val = kn.scratch_evaluate_model_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order11.arpa")
#print(val)
#kn.plot_precision_recall_curve("kenlm_prec_rec_curv_order2_20_main")
#accuracy = kn.paired_t_test([0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004,0.5743961352657004,0.5743961352657004],[0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609,0.5739130434782609])
#print("accuracy parametric ttest on kenln",accuracy)
#precision = kn.paired_t_test([0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,0.7420327232576317,0.7419880416193364,0.7419880416193364,0.7419880416193364],[0.7419880416193364,0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7419431944934916,0.7418981809590166,0.7419431944934916,0.7419431944934916])
#print("precision parametric ttest result on kenln",precision)
#f1 = kn.paired_t_test([0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,0.5355082179384458,0.5352446315786005,0.5352446315786005,0.5352446315786005],[0.5352446315786005,0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5349805637824815,0.5347160132298798,0.5349805637824815,0.5349805637824815])
#print("f1 parametric ttest result on kenlm",f1)
#print(kn.access_train_data_kenlm("scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/online/kenlm/build")) 


accuracy_wilconsin = kn.wilcon_t_test([0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798,0.5748792270531401,0.5743961352657004],[0.5743961352657004,0.5743961352657004,0.5743961352657004,0.5734299516908212,0.5739130434782609,0.5739130434782609,0.5739130434782609])
print("accuracy wilconxon test for kenlm ", accuracy_wilconsin)
precision_wilconsin = kn.wilcon_t_test([0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737,0.7420327232576317,0.7419880416193364],[0.7419880416193364,0.7419880416193364,0.7419880416193364,0.7418981809590166,0.7419431944934916,0.7419431944934916,0.7419431944934916])
print("precision wilconxon test for kenlm ", precision_wilconsin)
f1_wilconxon = kn.wilcon_t_test([0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169,0.5355082179384458,0.5352446315786005],[0.5352446315786005,0.5352446315786005,0.5352446315786005,0.5347160132298798,0.5349805637824815,0.5349805637824815,0.5349805637824815])
print("f1 wilconxon test for kenlm ", f1_wilconxon)
#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 7  --discount_fallback < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order7.arpa       
#cmake -DKENLM_MAX_ORDER=10 ..
#/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 20  --discount_fallback < /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt > /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/kelmn/arpas3/kenlmn_upd_order20.arpa