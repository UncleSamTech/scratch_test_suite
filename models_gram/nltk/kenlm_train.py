import os
#import kenlm
import sys
import nltk
import subprocess

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


kn = kenlm_train()




print(kn.access_train_data_kenlm("scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt","/mnt/c/Users/USER/Documents/model_train/online/kenlm/build")) 

        