import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class bi_lstm_scratch:

    def __init__(self):
        self.data = None
        self.token_list = []
        self.tokenizer = None
        self.input_sequences = []
        self.total_words = 0
        self.ne_input_sequences = []
        self.encompass = []

    
    def tokenize_data_inp_seq(self,file_name):
        
        with open(file_name,"r",encoding="utf-8") as rf:
            lines = rf.readlines()
            #qjg = self.quick_iterate(lines)
            max_len_ov = max([len(each_line) for each_line in lines])
            self.tokenizer = Tokenizer(oov_token='<oov>')
            self.tokenizer.fit_on_texts(lines)
            self.total_words = len(self.tokenizer.word_index) + 1
            for each_line in lines:
                each_line = each_line.strip()
                self.token_list = self.tokenizer.texts_to_sequences([each_line])[0]
                for i in range(1,len(self.token_list)):
                    ngram_seq = self.token_list[:i+1]
                    self.encompass.append(ngram_seq)
        return self.encompass,self.total_words
    
  
    
    def quick_iterate(self,list_words):
        word_lengths = {word: len(word) for word in list_words if isinstance(list_words,list) and len(list_words) > 0}
        max_word = max(word_lengths,key=word_lengths.get)
        max_count = word_lengths[max_word]

        max_word_dict = {max_word:max_count}
        return word_lengths, max_word_dict
    

    def pad_sequ(self,input_seq):
        
        
        max_seq_len = max([len(x) for x in input_seq])
        padded_in_seq = np.array(pad_sequences(input_seq,maxlen=max_seq_len,padding='pre'))
        return padded_in_seq,max_seq_len

    def prep_seq_labels(self,padded_seq,total_words):
        xs,labels = padded_seq[:,:-1],padded_seq[:,-1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        return xs,ys,labels
    
    def train_stand_alone(self,total_words,max_seq,xs,ys):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(total_words,100,input_shape=(max_seq-1,)))
        model.add(Bidirectional(LSTM(150)))
        model.add(Dense(total_words,activation='softmax'))
        adam = Adam(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
        history = model.fit(xs,ys,epochs=50,verbose=1)
        print(model.summary())
        
        return history

    def plot_graph(self,history,string_va):
        
        plt.plot(history.history[string_va])
        plt.xlabel("Epochs")
        plt.ylabel(string_va)
        plt.savefig(f"{string_va}_display.pdf")
        

    def consolidate_data(self,filepath):
        
        input_seq,total_words = self.tokenize_data_inp_seq(filepath)
        padd_seq,max_len = self.pad_sequ(input_seq)
        xs,ys,labels = self.prep_seq_labels(padd_seq,total_words)
        history = self.train_stand_alone(total_words,max_len,xs,ys)
        self.plot_graph(history,["accuracy","loss"])

        print(history)

    

cl_ob = bi_lstm_scratch()
#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt")
cl_ob.consolidate_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt")