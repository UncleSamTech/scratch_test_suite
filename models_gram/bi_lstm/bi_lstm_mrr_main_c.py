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
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import pickle
import time
from sklearn.utils.class_weight import compute_class_weight

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
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG") for line in lines]
            print("see lines:", lines)

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

    def consolidate_data_train(self,filepath,result_path,test_data,proj_number,model_name):
        input_seq,total_words,tokenizer = self.tokenize_data_inp_seq(filepath,result_path)
        padd_seq,max_len = self.pad_sequ(input_seq)
        #xs,ys,labels = self.prep_seq_labels(padd_seq,total_words)
        self.evaluate_bilstm_mrr_single(test_data,max_len,model_name,result_path,proj_number)
        #self.evaluate_bilstm_mrr_single(test_data,max_len,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/main_bilstm_scratch_model_150embedtime1_main_2.keras",result_path,proj_number)
       
        #self.train_model_five_runs(total_words,max_len,xs,ys,result_path,test_data,proj_number)
        #print(history)
        
        #self.train_model_again(model_name,result_path,xs,ys)

        #self.plot_graph("loss",result_path)

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


    def evaluate_bilstm(self,test_data,maxlen,model,result_path,proj_number,train_time):
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
            
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG") for line in lines]
            for i,line in enumerate(lines):
               
                line = line.strip()
                
                
                sentence_tokens = line.split(" ")
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1].lower()

                predicted_next_word = self.predict_token(context,tokenz,model,maxlen)
                
                
            
                if predicted_next_word is not None:
                    y_true.append(true_next_word)
                
                    y_pred.append(predicted_next_word)
                
               
                if i % 500 == 0:
                    print(f"Progress: {i} lines processed.")

        if not y_true or not y_pred:
            print("No valid predictions made.")
            return None, None, None, None
        
        end_time = time.time()
        time_spent = end_time - start_time
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        f1score = f1_score(y_true,y_pred,average="weighted")

        metrics_file = f"{result_path}bilstmmetrics_150embedtime1_{proj_number}_projects.txt"
        if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
            with open(metrics_file,"a") as fl:
                fl.write(f"accuracy,precision,recall,f1score,training_time,evaluation_time\n")
        with open(metrics_file,"a") as blm:
            blm.write(f"{accuracy},{precision},{recall},{f1score},{train_time},{time_spent:.2f}\n")
        
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
        

        # Run model training for 2 runs, with each run with a sampled data
      
        for run in range(1, 3):
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

            end_time = time.time()
            time_spent = end_time - start_time
            print(f"Run {run} complete. Training time: {time_spent:.2f} seconds")

            # Save the model and record training details
            #model_file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_{run}.keras"
            self.evaluate_bilstm_mrr(test_data,max_seq,model,result_path,proj_number,time_spent)
            #model.save(model_file_name)

    def predict_token_score(self, context, token, tokenz, model, maxlen):
        token_list = tokenz.texts_to_sequences([context])
        if not token_list or len(token_list[0]) == 0:
            return -1  # Assign low score for empty contexts

        token_value = token_list[0][-maxlen + 1:] + [tokenz.word_index.get(token, 0)]
        padded_in_seq = pad_sequences([token_value], maxlen=maxlen-1, padding="pre")
        padded_in_seq = tf.convert_to_tensor(padded_in_seq)

        prediction = model.predict(padded_in_seq)
        return prediction[0][-1]  # Score of the token  

    def evaluate_bilstm_mrr(self, test_data, maxlen, model, result_path, proj_number, train_time):
        tokenz = None
        
        with open(f"{result_path}tokenized_file_50embedtime1.pickle", "rb") as tk:
            tokenz = pickle.load(tk)

        vocab = list(tokenz.word_index.keys())  # Training vocabulary
        reciprocal_ranks = []

        start_time = time.time()
        with open(test_data, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)

            for i, line in enumerate(lines):
                line = line.strip().replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG")
                sentence_tokens = line.split(" ")
                context = " ".join(sentence_tokens[:-1])  # Exclude last word
                true_next_word = sentence_tokens[-1].lower()

                scores = []
                for token in vocab:
                    context_score = self.predict_token_score(context, token, tokenz, model, maxlen)
                    scores.append((context_score, token))

                # Sort scores in descending order
                scores.sort(reverse=True, key=lambda x: x[0])

                # Extract top predictions
                top_predictions = [t[1] for t in scores[:10]]

                # Calculate reciprocal rank
                if true_next_word in top_predictions:
                    rank = top_predictions.index(true_next_word) + 1
                    reciprocal_ranks.append(1 / rank)
                else:
                    reciprocal_ranks.append(0)

                if i % 500 == 0:
                    print(f"Progress: {i} lines processed.")

        # Mean Reciprocal Rank
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

        end_time = time.time()
        time_spent = end_time - start_time

        metrics_file = f"{result_path}bilstm_mrr_metrics_{proj_number}.txt"
        if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
            with open(metrics_file, "a") as fl:
                fl.write("MRR,Training_Time,Evaluation_Time\n")
        with open(metrics_file, "a") as blm:
            blm.write(f"{mrr},{train_time},{time_spent:.2f}\n")

        print(f"MRR: {mrr}")
        return mrr  
            
    def evaluate_bilstm_mrr_single(self, test_data, maxlen, model, result_path, proj_number):
        tokenz = None
        loaded_model = load_model(model,compile=False)
        tokenizer_path = os.path.join(result_path, "tokenized_file_50embedtime1.pickle")
        with open(tokenizer_path, "rb") as tk:
          tokenz = pickle.load(tk)

        vocab = list(tokenz.word_index.keys())  # Training vocabulary
        print(f"vocabulary of 10 projects has a total words of {len(vocab)}  and they are : {vocab}")
        reciprocal_ranks = []

        start_time = time.time()
        with open(test_data, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
            lines = [line.replace("_", "UNDERSCORE").replace(">", "RIGHTANG").replace("<", "LEFTANG") for line in lines]
            for i, line in enumerate(lines):
                if not line.strip():
                    continue

                sentence_tokens = line.split(" ")
                if len(sentence_tokens) < 2:
                    continue
                
                #sentence_tokens = line.split(" ")
                context = " ".join(sentence_tokens[:-1])  # Exclude last word
                true_next_word = sentence_tokens[-1].lower()

                scores = []
                for token in vocab:
                    context_score = self.predict_token_score(context, token, tokenz,loaded_model, maxlen)
                    scores.append((context_score, token))

                    # Sort scores in descending order
                    scores.sort(reverse=True, key=lambda x: x[0])

                    # Extract top predictions
                    top_predictions = [t[1] for t in scores[:10]]

                    # Calculate reciprocal rank
                    if true_next_word in top_predictions:
                        rank = top_predictions.index(true_next_word) + 1
                        reciprocal_ranks.append(1 / rank)
                    else:
                        reciprocal_ranks.append(0)

                    if i % 1000 == 0:
                        print(f"Progress: {i}/{len(lines)} lines processed.")

        # Mean Reciprocal Rank
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

        time_spent = time.time() - start_time

        metrics_file = os.path.join(result_path, f"bilstm_mrr_metrics_{proj_number}.txt")
        os.makedirs(result_path, exist_ok=True)  # Ensure path exists
        with open(metrics_file, "a") as blm:
          if os.path.getsize(metrics_file) == 0:
              blm.write("MRR,Evaluation_Time\n")
          blm.write(f"{mrr},{time_spent:.2f}\n")

        print(f"MRR: {mrr}")
        return mrr  

cl_ob = bi_lstm_scratch()
#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt")
#cl_ob.consolidate_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","bilstm_scratch_model_100embedtime2.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/bi_lstm/results/results2/")

#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_00.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_portion/")

#cl_ob.evaluate_bilstm_mrr_single("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt",39,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/main_bilstm_scratch_model_150embedtime1_main_2.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/","10")
cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_projects_mrr/","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_1.txt","10","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/main_bilstm_scratch_model_150embedtime1_main_2.keras")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_50_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_100_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_100/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_150_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_150/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_500_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_500/")


#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_test_data_10.txt","bilstm_scratch_model_50embedtime1.keras","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/bi_lstm/results_local/")
#cl_ob.plot_graph("loss")
#cl_ob.evaluate_bilstm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt",39,"main_bilstm_scratch_model_150embedtime1_main_4.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/")
#cl_ob.predict_next_token_bilstm("event_whenflagclicked control_forever BodyBlock control_create_clone_of")