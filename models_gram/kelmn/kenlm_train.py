import os
#import kenlm
import sys
import nltk
import subprocess

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
                #val = reasp.encode('utf-8')
                command = f"{command_link} {reasp}"
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
kn = kenlm_train()




#print(kn.access_train_data_kenlm("/mnt/c/Users/USER/Documents/model_train/scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/scratch_test_suite/online/kenlm/build/bin/lmplz -o 5 > an_kenlm.arpa")) 

print(kn.access_train_data_kenlm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90_check.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/online/kenlm/build/bin/lmplz -o 5 > an_kenlm.arpa")) 
     