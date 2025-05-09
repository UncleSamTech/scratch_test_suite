import pandas as pd
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix
import pickle
import time
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import re 

class bi_lstm_scratch:

    def __init__(self):
        self.data = None
        self.token_list = []
        self.tokenizer = None
        self.input_sequences = []
        self.total_words = 0
        self.ne_input_sequences = []
        self.encompass = []
        self.model = keras.Sequential()

    
    def tokenize_data_inp_seq(self, file_name, result_path):
        with open(file_name, "r", encoding="utf-8") as rf:
            lines = rf.readlines()
            #shuffle trainset every run
            random.shuffle(lines)
            # Replace specific characters
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            #print("see lines:", lines)

            # Initialize and fit the tokenizer
            self.tokenizer = Tokenizer(oov_token='<oov>')
            self.tokenizer.fit_on_texts(lines)

            # Save the tokenizer
            with open(f"{result_path}tokenized_file_50embedtime1.pickle", "wb") as tk:
                pickle.dump(self.tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

            # Define total_words based on the tokenizer
            self.total_words = len(self.tokenizer.word_index) + 1  # +1 to account for <oov>
            
            print(f"Total words (vocabulary size): {self.total_words}")

            # Generate token sequences (ngrams)
            self.encompass = []
            max_index = 0  # Track max token index to verify alignment with `total_words`
            for each_line in lines:
                each_line = each_line.strip()
                self.token_list = self.tokenizer.texts_to_sequences([each_line])[0]
                #max_index = max(max_index, max(self.token_list, default=0))  # Update max_index
                for i in range(1, len(self.token_list)):
                    ngram_seq = self.token_list[:i + 1]
                    self.encompass.append(ngram_seq)

            # Verify that total_words aligns with max index in token_list
            # if max_index >= self.total_words:
            #     print(f"Adjusting total_words to cover max token index: {max_index}")
            #     self.total_words = max_index + 1  # Update total_words if needed

            #print(f"First stage complete with encompass: {self.encompass}, total_words: {self.total_words}")
            return self.encompass, self.total_words, self.tokenizer
    
  
    
    def quick_iterate(self,list_words):
        word_lengths = {word: len(word) for word in list_words if isinstance(list_words,list) and len(list_words) > 0}
        max_word = max(word_lengths,key=word_lengths.get)
        max_count = word_lengths[max_word]

        max_word_dict = {max_word:max_count}
        return word_lengths, max_word_dict
    

    def pad_sequ(self,input_seq):
        
        
        max_seq_len = max([len(x) for x in input_seq])
        padded_in_seq = np.array(pad_sequences(input_seq,maxlen=max_seq_len,padding='pre'))
        #print("input shape training  ", padded_in_seq.shape)
        return padded_in_seq,max_seq_len

    def prep_seq_labels(self,padded_seq,total_words):
        xs,labels = padded_seq[:,:-1],padded_seq[:,-1]

        max_label_index = np.max(labels)
        if max_label_index >= total_words:
            print(f"Adjusting total_words from {total_words} to {max_label_index + 1} based on labels.")
            total_words = max_label_index + 1
        
        # Ensure labels do not exceed the total words range
        if np.any(labels >= total_words):
            raise ValueError(f"Labels contain indices >= total_words: {np.max(labels)} >= {total_words}")
    
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        return xs, ys, labels
        #ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        #return xs,ys,labels
    
    def train_stand_alone(self,total_words,max_seq,xs,ys,result_path):
        print(tf.__version__)
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Default GPU device: {gpus[0]}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu,True)
                print(f"Default GPU device : {tf.test.gpu_device_name()}")
                start_time = time.time()
                with tf.device('/GPU:0'):
                    model = Sequential()
                    model.add(Embedding(total_words,100,input_shape=(max_seq-1,)))
                    model.add(Bidirectional(LSTM(150)))
                    model.add(Dense(total_words,activation='softmax'))
                    adam = Adam(learning_rate=0.01)
                    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
                    history = model.fit(xs,ys,epochs=50,verbose=1)

                    with open(f"{result_path}main_historyrec_150embedtime1.pickle","wb") as hs:
                        pickle.dump(history,hs)

                    print(model.summary())
                    end_time = time.time()
                    time_spent = end_time - start_time
                    file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1.keras"
                    if os.path.exists(file_name):
                        os.remove(file_name)

                    with open(f"{result_path}main_seqlen_150embedtime1.txt","a") as se:
                        se.write(f"sequence length {max_seq} training time {time_spent:.2f} seconds \n")
                

                    model.save(file_name)
                    
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("Please install GPU version of TF")
            start_time = time.time()
            
            model = Sequential()
            model.add(Embedding(total_words,100,input_shape=(max_seq-1,)))
            model.add(Bidirectional(LSTM(150)))
            model.add(Dense(total_words,activation='softmax'))
            adam = Adam(learning_rate=0.01)
            model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
            history = model.fit(xs,ys,epochs=50,verbose=1)
            
            with open(f"{result_path}main_historyrec_150embedtime1.pickle","wb") as hs:
                    pickle.dump(history,hs)

            print(model.summary())
            end_time = time.time()
            time_spent = end_time - start_time
            file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1.keras"
            if os.path.exists(file_name):
                os.remove(file_name)

            with open(f"{result_path}main_seqlen_150embedtime1.txt","a") as se:
                se.write(f"sequence length {max_seq} training time {time_spent:.2f} seconds \n")

            model.save(file_name)
            #print("model weight",model.get_weights())

            return history,model
        

    def plot_graph(self,string_va,result_path):

        with open(f"{result_path}main_historyrec_150embedtime5.pickle","rb") as rh:
            val = pickle.load(rh)
        
            plt.plot(val.history[string_va])
        
        '''
        loss = [1.0493,0.9448,0.9294,0.9223,0.9198,0.9192,0.9217,0.9140,0.9241,0.9218,0.9215,0.9208,
                0.9187, 0.9206,0.9247,0.9275,0.9372,0.9325,0.9324,0.9357,0.9451, 0.9523,0.9438,0.9509,
                0.9501,0.9472,0.9444,0.9599,0.9532,0.9533,0.9520,0.9503,0.9522,0.9554,0.9560,0.9576,0.9481,
                0.9518,0.9568,0.9458,0.9449,0.9488,0.9444,0.9530,0.9678,0.9587,0.9527,0.9536,0.9523,0.9499]
        '''
        epochs = list(range(1,51))
        #plt.plot(epochs,loss)
        plt.xlabel("Epochs")
        plt.ylabel(string_va)
        #plt.show()
        plt.savefig(f"{result_path}{string_va}bilstm_150embedtime5_quick.pdf")

        

    def train_model_again(self,model_name,result_path,xs,ys):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Default GPU device: {gpus[0]}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {tf.test.gpu_device_name()}")

            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
                return

        else:
            print("No GPU available. Running on CPU.")
        model_name_comp = f"{result_path}{model_name}"
        
        loaded_model = load_model(model_name_comp,compile=True)
        # Reduce learning rate when a metric has stopped improving
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        history = loaded_model.fit(xs,ys,epochs=50,verbose=1,callbacks=[lr_scheduler,early_stopping])

        file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_5.keras"
        
        loaded_model.save(file_name)

        with open(f"{result_path}main_historyrec_150embedtime5.pickle","wb") as hs:
            pickle.dump(history,hs)
                        

    def consolidate_data(self,filepath,testfile,model_path,result_path):
        
        input_seq,total_words,tokenizer = self.tokenize_data_inp_seq(filepath,result_path)
        padd_seq,max_len = self.pad_sequ(input_seq)
        xs,ys,labels = self.prep_seq_labels(padd_seq,total_words)
        #history_again = self.train_model_again(model_path,result_path,xs,ys)
        #history,model = self.train_stand_alone(total_words,max_len,xs,ys,result_path)

        
        val = self.evaluate_bilstm(testfile,max_len,model_path,result_path)
        #print(history)
        #self.plot_graph("accuracy",result_path)
        #self.plot_graph("loss",result_path)
        #val = self.predict_word("event_whenflagclicked control_forever",model,2,max_len,tokenizer)
        #print(val)
        
        #print(model)
        #return val

    def consolidate_data_train(self,filepath,result_path,test_data,proj_number):
        input_seq,total_words,tokenizer = self.tokenize_data_inp_seq(filepath,result_path)
        padd_seq,max_len = self.pad_sequ(input_seq)
        xs,ys,labels = self.prep_seq_labels(padd_seq,total_words)
        
       
        self.train_model_five_runs(total_words,max_len,xs,ys,result_path,test_data,proj_number)
        av = ["main_bilstm_scratch_model_150embedtime1_main_sample_project10_run5.keras"]

        all_models = sorted([files for files in os.listdir(result_path) if files.endswith(".keras") and files in av])
        print(all_models)
        
        if all_models:
            for model in all_models:
                
                match = re.search(r"run(\d+)",model.strip())

                if match:
                    run = match.group(1)
                    model = os.path.join(result_path,model).strip()

                    self.evaluate_bilstm_in_order(test_data,max_len,model,result_path,proj_number,"0",run)

    def predict_word(self,seed_text,model,next_words_count,max_seq_len,tokenize_var):
        
        for _ in range(next_words_count):
            token_list = tokenize_var.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],maxlen=max_seq_len - 1,padding='pre')
            #print("tokenlist",token_list)
            predicted = model.predict(token_list,verbose=0)
            #print(predicted)
            output_word = ""
            for word,index in tokenize_var.word_index.items():
                #print(f'index {index} {type(index)}')
                #print("word ", word )
                if index == predicted.any():
                    output_word = word
                    break
            seed_text += " " + output_word
        print(seed_text)
        return seed_text


    def evaluate_bilstm(self,test_data,maxlen,model,result_path,proj_number,train_time,run):
        y_true = []
        y_pred = []
        tokenz = None
        #loaded_model = load_model(f"{model_path}",compile=False)
        with open(f"{result_path}tokenized_file_50embedtime1.pickle","rb") as tk:
            tokenz = pickle.load(tk)
            
        
        # Start the evaluation timer
        start_time = time.time()

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            for i,line in enumerate(lines):
               
                line = line.strip()
                
                
                sentence_tokens = line.split(" ")
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]

                predicted_next_word = self.predict_token(context,tokenz,model,maxlen)
                
                
            
                if predicted_next_word is not None:
                    y_true.append(true_next_word)
                
                    y_pred.append(predicted_next_word)
                
               
                if i % 500 == 0:
                    print(f"Progress: {i} lines processed.")

        if not y_true or not y_pred:
            print("No valid predictions made.")
            return None, None, None, None
        
        #self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        end_time = time.time()
        time_spent = end_time - start_time
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)

        metrics_file = f"{result_path}bilstmmetrics_150embedtime1_{proj_number}_projects.txt"
        if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
            with open(metrics_file,"a") as fl:
                fl.write(f"accuracy,precision,recall,f1score,training_time,evaluation_time\n")
        with open(metrics_file,"a") as blm:
            blm.write(f"{accuracy},{precision},{recall},{f1score},{train_time},{time_spent:.2f}\n")

        self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        return accuracy,precision,recall,f1score
    def predict_token_score_upd(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.

        Args:
            context (str): Input context for prediction.
            tokenz (Tokenizer): Tokenizer object with vocabulary and word index.
            model (tf.keras.Model): Trained model for next token prediction.
            maxlen (int): Maximum length of input sequences for the model.

        Returns:
            tuple: Predicted next token and a list of the top 10 tokens with their scores.
        """
        # Convert the context into a sequence of token indices
        token_list = tokenz.texts_to_sequences([context])
        vocab = list(tokenz.word_index.keys())
        max_prob_tokens = {}

        # Check if the context is empty or invalid
        if not token_list or len(token_list[0]) == 0:
            return -1, []  # Return low score and empty top 10 for invalid context

        # Iterate through the entire vocabulary
        for each_token in vocab:
            # Prepare the input sequence with the current token appended
            token_value = token_list[0][-maxlen + 1:] + [tokenz.word_index.get(each_token, 0)]
            padded_in_seq = pad_sequences([token_value], maxlen=maxlen-1, padding="pre")
            padded_in_seq = tf.convert_to_tensor(padded_in_seq)

            # Get the model's prediction probabilities
            prediction = model.predict(padded_in_seq, verbose=0)[0]
            
            # Store the score for the token (assuming prediction gives probabilities per token in vocab size)
            token_index = tokenz.word_index.get(each_token, 0)
            max_prob_tokens[each_token] = prediction[token_index]

        # Determine the most probable token
        predicted_next_token = max(max_prob_tokens, key=max_prob_tokens.get)

        # Extract the top 10 tokens and their scores
        top_10_tokens_scores = sorted(max_prob_tokens.items(), key=lambda item: item[1], reverse=True)[:10]

        return predicted_next_token, top_10_tokens_scores


    def predict_token_score_upd2(self, context, tokenz, model, maxlen):
        """
        Predicts the next token based on the given context and scores each token in the vocabulary.

        Args:
            context (str): Input context for prediction.
            tokenz (Tokenizer): Tokenizer object with vocabulary and word index.
            model (tf.keras.Model): Trained model for next token prediction.
            maxlen (int): Maximum length of input sequences for the model.

        Returns:
            tuple: Predicted next token and a list of the top 10 tokens with their scores.
        """
        # Convert the context into a sequence of token indices
        token_list = tokenz.texts_to_sequences([context])
        vocab = list(tokenz.word_index.keys())
        vocab_size = len(vocab)
        max_prob_tokens = np.zeros(vocab_size)  # Preallocate array for token probabilities

        # Check if the context is empty or invalid
        if not token_list or len(token_list[0]) == 0:
            return -1, []  # Return low score and empty top 10 for invalid context

        # Iterate through the entire vocabulary
        for idx, each_token in enumerate(vocab):
            # Prepare the input sequence with the current token appended
            token_value = token_list[0][-maxlen + 1:] + [tokenz.word_index.get(each_token, 0)]
            padded_in_seq = pad_sequences([token_value], maxlen=maxlen-1, padding="pre")
            padded_in_seq = tf.convert_to_tensor(padded_in_seq)

            # Get the model's prediction probabilities
            prediction = model.predict(padded_in_seq, verbose=0)[0]
            
            # Store the score for the token
            max_prob_tokens[idx] = prediction[tokenz.word_index.get(each_token, 0)]

        # Determine the most probable token using np.argmax
        predicted_index = np.argmax(max_prob_tokens)
        predicted_next_token = vocab[predicted_index]

        # Extract the top 10 tokens and their scores
        top_10_indices = np.argsort(max_prob_tokens)[-10:][::-1]
        top_10_tokens_scores = [(vocab[i], max_prob_tokens[i]) for i in top_10_indices]

        return predicted_next_token, top_10_tokens_scores


    def check_available_rank(self,list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                return rank
        return rank

    def evaluate_bilstm_in_order_upd_norun(self,test_data,maxlen,model,result_path,proj_number,new_path):
        y_true = []
        y_pred = []
        tokenz = None
        loaded_model = load_model(f"{model}",compile=False)
        with open(f"{result_path}tokenized_file_50embedtime1.pickle","rb") as tk:
            tokenz = pickle.load(tk)
            
        
        # Start the evaluation timer
        start_time = time.time()

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            #lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split(" ")
                if len(sentence_tokens) < 2:
                    continue
                
                # evaluate each token in order starting from the second token
                for idx in range(1,len(sentence_tokens)):

                    context = ' '.join(sentence_tokens[:idx])  
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word,top_10_tokens = self.predict_token_score_upd2(context,tokenz,loaded_model,maxlen)
                    rank = self.check_available_rank(top_10_tokens,true_next_word)
                    investig_path = f"{new_path}/bilstm_investigate_{proj_number}_3.txt"
                    if not os.path.exists(investig_path) or os.path.getsize(investig_path) == 0:
                        with open(investig_path,"a") as ip:
                            ip.write(f"query,expected,answer,rank,correct\n")
                    with open(investig_path,"a") as inv_path_file:
                        inv_path_file.write(f"{context.strip()},{true_next_word.strip()},{predicted_next_word},{rank},{1 if true_next_word.strip() == predicted_next_word else 0}\n")

                

                
        end_time = time.time() - start_time
        print(f"duration for bilstm {proj_number} projects sample is {end_time}") 
        #             if predicted_next_word is not None:
        #                 y_true.append(true_next_word)
                
        #                 y_pred.append(predicted_next_word)

        # if not y_true or not y_pred:
        #     print("No valid predictions made.")
        #     return None, None, None, None
        
        # #self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        # end_time = time.time()
        # time_spent = end_time - start_time
        # accuracy = accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        # recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        # f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)

        # metrics_file = f"{result_path}bilstmmetrics_150embedtime1_{proj_number}_projects.txt"
        # if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
        #     with open(metrics_file,"a") as fl:
        #         fl.write(f"accuracy,precision,recall,f1score,training_time,evaluation_time\n")
        # with open(metrics_file,"a") as blm:
        #     blm.write(f"{accuracy},{precision},{recall},{f1score},{train_time},{time_spent:.2f}\n")

        # self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        # return accuracy,precision,recall,f1score



    def evaluate_bilstm_in_order(self,test_data,maxlen,model,result_path,proj_number,train_time,run):
        y_true = []
        y_pred = []
        tokenz = None
        loaded_model = load_model(f"{model}",compile=False)
        with open(f"{result_path}tokenized_file_50embedtime1.pickle","rb") as tk:
            tokenz = pickle.load(tk)
            
        
        # Start the evaluation timer
        start_time = time.time()

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG").lower() for line in lines]
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split(" ")
                if len(sentence_tokens) < 2:
                    continue
                
                # evaluate each token in order starting from the second token
                for idx in range(1,len(sentence_tokens)):

                    context = ' '.join(sentence_tokens[:idx])  
                    true_next_word = sentence_tokens[idx]

                    predicted_next_word = self.predict_token(context,tokenz,loaded_model,maxlen)
                
                
            
                    if predicted_next_word is not None:
                        y_true.append(true_next_word)
                
                        y_pred.append(predicted_next_word)

        if not y_true or not y_pred:
            print("No valid predictions made.")
            return None, None, None, None
        
        #self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        end_time = time.time()
        time_spent = end_time - start_time
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro',zero_division=0)
        f1score = f1_score(y_true,y_pred,average="macro",zero_division=0)

        metrics_file = f"{result_path}bilstmmetrics_150embedtime1_{proj_number}_projects.txt"
        if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
            with open(metrics_file,"a") as fl:
                fl.write(f"accuracy,precision,recall,f1score,training_time,evaluation_time\n")
        with open(metrics_file,"a") as blm:
            blm.write(f"{accuracy},{precision},{recall},{f1score},{train_time},{time_spent:.2f}\n")

        self.compute_confusion_matrix(y_true,y_pred,result_path,proj_number,run)
        
        return accuracy,precision,recall,f1score

    
    def predict_next_token_bilstm(self,context,maxseqlen,model_name,result_path):
        token_list = None
        token_value = None
        gpu = tf.config.list_physical_devices('GPU')
        output_word = ""
        
        with open(f"{result_path}tokenized_file_50embedtime1.pickle","rb") as tk:
            tokenz = pickle.load(tk)
            context = context.strip()

            token_list = tokenz.texts_to_sequences([context])
            if not token_list or len(token_list[0]) == 0:
                print("Empty token list, unable to predict token.")
                return None
            token_value = token_list[0]
            if gpu:
                print(f"Default GPU device : {gpu[0].name}")
            
            
                padded_in_seq = pad_sequences([token_value],maxlen=maxseqlen-1,padding='pre')
                
                try:
                    load_mod = load_model(f"{result_path}{model_name}",compile=False)
                except OSError as e:
                    
                    return None
                predicted = load_mod.predict(padded_in_seq)
                

                pred_token_index = np.argmax(predicted,axis=-1)
        
     
                
                for token,index in tokenz.word_index.items():
                    if index == pred_token_index:
                        output_word = token
                        print(output_word)
                        break
                            
                return output_word
        
            else:   
                padded_in_seq = np.array(pad_sequences([token_value],maxlen=maxseqlen-1,padding='pre',truncating='pre')) 
                try:
                    load_mod = load_model(f"{result_path}{model_name}",compile=False)
                except OSError as e:
                    print(f"Error loading model: {e}")
                    return None 
                predicted = load_mod.predict(padded_in_seq)
                
                
                pred_token_index = np.argmax(predicted,axis=-1)
                
                #print("index",pred_token_index)

                for word,index in tokenz.word_index.items():
                    if index == pred_token_index:
                        output_word = word
                        print(output_word)
                        break
                return output_word
       

    def predict_token(self,context, tokenz, load_mod, maxseqlen):
        token_list = None
        token_value = None
        output_word = ""
    
        
        # Tokenize context
        context = context.strip()
        #context = context.replace("_","UNDERSCORE")
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            print("Empty token list, unable to predict token.")
            return None
    
        token_value = token_list[0]
        padded_in_seq = pad_sequences([token_value], maxlen=maxseqlen - 1, padding='pre')

        # Ensure input is a tensor with consistent shape
        padded_in_seq = tf.convert_to_tensor(padded_in_seq)

        # Predict the next token
        predicted = load_mod.predict(padded_in_seq)

        # Retrieve the predicted token
        pred_token_index = np.argmax(predicted, axis=-1)
        for token, index in tokenz.word_index.items():
            if index == pred_token_index:
                output_word = token
                print(output_word)
                break
        #output_word  = output_word.replace("UNDERSCORE","_")
        return output_word

    def load_trained_model(self,model_name) :
        with open(model_name,"rb") as f:
            self.loaded_scratch_model = pickle.load(f)
        return self.loaded_scratch_model

    

    def view_model_summary(self,model_path):
        ld = load_model(model_path)
        ld.summary()


    def train_model_five_runs(self, total_words, max_seq, xs, ys, result_path,test_data,proj_number):
        print(tf.__version__)
        print("max length",max_seq)
        
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Default GPU device: {gpus[0]}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {tf.test.gpu_device_name()}")

            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
                return

        else:
            print("No GPU available. Running on CPU.")

        
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        

        # Run model training for 5 runs, with each run with a sampled data
      
        for run in range(1, 6):
            print(f"\nStarting run {run}...\n")
            start_time = time.time()

           
            
            model = Sequential([
                Embedding(total_words, 100, input_shape=(max_seq - 1,)),
                Bidirectional(LSTM(150)),
                Dense(total_words, activation='softmax')
                ])
            adam = Adam(learning_rate=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            

            
            # Fit the model
            history = model.fit(xs, ys, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

            # Save the history
            with open(f"{result_path}main_historyrec_150embedtime{run}.pickle", "wb") as hs:
                pickle.dump(history.history, hs)
            
            #save the model for every run
            file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_sample_project{proj_number}_run{run}.keras"
            
            if os.path.exists(file_name):
                os.remove(file_name)
            model.save(file_name)

            end_time = time.time()
            time_spent = end_time - start_time
            print(f"Run {run} complete. Training time: {time_spent:.2f} seconds")

            # Save the model and record training details
            #model_file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_{run}.keras"
            self.evaluate_bilstm_in_order(test_data,max_seq,model,result_path,proj_number,time_spent,run)
            #model.save(model_file_name)

            
    def compute_confusion_matrix(self, y_true, y_pred, result_path, proj_number,run,top_k=10):
        labels = np.unique(np.concatenate((y_true, y_pred)))  # Get unique labels
        id2label = {i: str(label) for i, label in enumerate(labels)}  # Map indices to labels
        label2id = {v: k for k, v in id2label.items()}  # Reverse mapping (if needed)

        # Compute confusion matrix
        print("\nComputing Confusion Matrix...")
    
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        num_classes = conf_matrix.shape[0]
        print(f" number of classes {num_classes}")
        metrics = {id2label[i]:{"TP":0,"FP":0,"FN":0,"TN":0} for i in range(num_classes)}
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for i in range(num_classes):
            TP = conf_matrix[i,i]
            FP = np.sum(conf_matrix[:,i]) - TP
            FN = np.sum(conf_matrix[i, :]) - TP
            TN = np.sum(conf_matrix) - (TP + FP + FN)

            label = id2label[i]
            metrics[label]["TP"] = TP
            metrics[label]["FP"] = FP
            metrics[label]["FN"] = FN
            metrics[label]["TN"] = TN

            total_tp += TP
            total_fp += FP
            total_fn += FN
            total_tn += TN

        # Write metrics to file and print
        with open(f"{result_path}tp_fp_fn_tn_label_val_{proj_number}_{run}.txt", "w") as af:
            af.write("Class,TP,FP,FN,TN\n")  # Header
            for label, values in metrics.items():
                #print(f"Label {label}: TP={values['TP']}, FP={values['FP']}, FN={values['FN']}, TN={values['TN']}")
                af.write(f"{label},{values['TP']},{values['FP']},{values['FN']},{values['TN']}\n")

        # Print total metrics
        with open(f"{result_path}total_results_bilstm_tp_tn_fp_fn_{proj_number}_{run}.txt","w") as tot:
          tot.write("total_tn,total_fp,total_fn,total_tp\n")
          tot.write(f"{total_tn},{total_fp},{total_fn},{total_tp}")
        print(f"\nTotal TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        conf_matrix = np.array([[total_tp, total_fn],
                            [total_fp, total_tn]])

        # Plotting the confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Positive', 'Predicted Negative'], 
                yticklabels=['Actual Positive', 'Actual Negative'])

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        #plt.show()
    
        # # Get the unique class labels in sorted order (this will be used for indexing)
        # unique_classes = np.unique(np.concatenate((y_true, y_pred)))  # Combine y_true and y_pred to cover all classes
    
        # # Determine the top-k most frequent classes based on y_true
        # class_counts = pd.Series(y_true).value_counts().head(top_k).index
    
        # # Map the class labels to indices based on the sorted unique classes
        # class_indices = [np.where(unique_classes == label)[0][0] for label in class_counts]
    
        # # Use np.ix_ to index into the confusion matrix
        # filtered_conf_matrix = conf_matrix[np.ix_(class_indices, class_indices)]
    
        # # Optional: Save confusion matrix as a heatmap
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(filtered_conf_matrix, annot=True, fmt='d', cmap='Blues',
        #         xticklabels=class_counts, yticklabels=class_counts)
        
        # # Rotate x-axis labels to avoid overlap
        # plt.xticks(rotation=45, ha='right')  # Rotate labels and align them to the right
        # plt.yticks(rotation=0)  # Keep y-axis labels as they are

        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.title(f'Confusion Matrix (Top {top_k} Classes)')
        # # Adjust layout to make sure everything fits
        # plt.tight_layout()
        plt.savefig(f"{result_path}confusion_matrix_run_an_bilstm_{proj_number}_{run}.pdf")
        plt.close()        

cl_ob = bi_lstm_scratch()
#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt")
#cl_ob.consolidate_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","bilstm_scratch_model_100embedtime2.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/bi_lstm/results/results2/")

#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_00.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_portion/")

#cl_ob.view_model_summary("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50_projects_conf/main_bilstm_scratch_model_150embedtime1_main_sample_project50_run1.keras")
cl_ob.evaluate_bilstm_in_order_upd_norun("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_44_projects_model_test_kenlm_part_ac.txt",27,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50_projects_conf/main_bilstm_scratch_model_150embedtime1_main_sample_project50_run2.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50_projects_conf/",50,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50_projects_conf/new_metrics")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_projects_conf/","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt","10")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_50_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_100_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_100/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_150_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_150/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_500_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_500/")


#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_test_data_10.txt","bilstm_scratch_model_50embedtime1.keras","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/bi_lstm/results_local/")
#cl_ob.plot_graph("loss")
#cl_ob.evaluate_bilstm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt",39,"main_bilstm_scratch_model_150embedtime1_main_4.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/")
#cl_ob.predict_next_token_bilstm("event_whenflagclicked control_forever BodyBlock control_create_clone_of")