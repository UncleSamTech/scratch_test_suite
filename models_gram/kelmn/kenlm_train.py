import os
#import kenlm
import sys
import nltk
import subprocess
import matplotlib.pyplot as plt
#import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_curve,f1_score

class kenlm_train:

    def __init__(self):
        self.result = []
        self.tokenized_data = " "

    def tokenize_kenlm(self,train_data,command_link):
        for line in train_data:
            
            for sentence in nltk.sent_tokenize(line):
                #print(f' type {type(sentence)} value {sentence}')
                
                #sentence = sentence.split()
                #val = list(sentence)
                #print("list",val)
                sent_list = [sentence]
                resp = self.slice_from_start(sent_list)
                
                token_sentlist = nltk.word_tokenize(resp)
                new_val = " ".join(token_sentlist).lower()
                reasp = " " + new_val
                print(reasp)
                val = reasp.encode('utf-8')
                command = f"{command_link} {val}"
                module_train = subprocess.run(command,shell=True)

        return module_train.stdout
        #return self.tokenized_data

    
    def access_train_data_kenlm(self,file_path,command_link):
        if os.path.isfile(file_path):
            with open(file_path,"r") as each_sentence:
                each_line = each_sentence.readlines()
                val = self.tokenize_kenlm(each_line,command_link)
                print(val + " ")

    def slice_from_start(self,string_val):
        val = ''
        if string_val is not None:
            try:
                val = " ".join(string_val)
            except:
                val = ''
            keywords = ["event_","control_","procedures_"]
            if len(val) > 0:
                start_position = min((val.find(keyword) for keyword in keywords if keyword in val), default=-1)
                if start_position != -1:
                    extr_text = val[start_position:]
            
                    return extr_text

    def plot_precision_recall_curve(self,plot_name):

        Accuracy = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        Precision = [0.7615713716522035,0.7592280310176857,0.7622305260526447,0.7582152779712141,0.7420772403226737]
        Recall = [0.5507246376811594,0.5685990338164251,0.5681159420289855,0.5777777777777777,0.5753623188405798]
        F1 = [0.5098150651292701,0.5294639842492523,0.5345024599207917,0.5397726754613954,0.5357713241774169]
        Ngrams = [2,3,4,5,6]

        
        
        plt.plot(Ngrams, Accuracy, label = "Accuracy",color="blue",marker="o",linestyle="-")
        plt.plot(Ngrams, Precision, label = "Precision",color="red")
        plt.plot(Ngrams, Recall, label = "Recall",color="yellow")
        plt.plot(Ngrams, F1, label = "F1",color="green")
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Kenlm_Model Scores vs N-Gram Orders for replaced tokens')
        plt.legend()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}.pdf')
        #plt.show()

kn = kenlm_train()




#print(kn.access_train_data_kenlm("/mnt/c/Users/USER/Documents/model_train/scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/scratch_test_suite/online/kenlm/build/bin/lmplz -o 5 > an_kenlm.arpa")) 

print(kn.access_train_data_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90_check.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 5 > an_kenlm.arpa")) 
kn.plot_precision_recall_curve("kenlm_prec_rec_curv_order2_6.pdf")  