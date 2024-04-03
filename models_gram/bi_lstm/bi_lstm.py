import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class bi_lstm_scratch:

    def __init__(self):
        self.data = None

    
    def tokenize_data(self,file_name):
        total_words = 0
        with open(file_name,"r",encoding="utf-8") as rf:
            lines = rf.readlines()
            for each_line in lines:
                each_line = each_line.strip()
                tokenizer = Tokenizer(oov_token='<oov>')
                tokenizer.fit_on_texts(each_line)
                total_words += len(tokenizer.word_index) + 1

            print("Total number of words :" , total_words)
            print("Word: ID")
            print("------------")
            print("<oov>: ", tokenizer.word_index['<oov>'])
            #print("event_whenflagclicked: ", tokenizer.word_index['event_whenflagclicked'])
            #print("control_forever: ", tokenizer.word_index['control_forever'])
            #print("Consumption: ", tokenizer.word_index['consumption'])


cl_ob = bi_lstm_scratch()
#cl_ob.tokenize_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/scratch_train_data_90.txt")
cl_ob.tokenize_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt")