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

    
    def tokenize_data_inp_seq(self,file_name,result_path):
        
        with open(file_name,"r",encoding="utf-8") as rf:
            lines = rf.readlines()
            lines = [line.replace("_","UNDERSCORE") for line in lines]
            lines = [line.replace(">","RIGHTANG") for line in lines]
            lines = [line.replace("<","LEFTANG") for line in lines]
            print("see lines" ,lines)
            #qjg = self.quick_iterate(lines)
            max_len_ov = max([len(each_line) for each_line in lines])
            self.tokenizer = Tokenizer(oov_token='<oov>')
            self.tokenizer.fit_on_texts(lines)

            with open(f"{result_path}tokenized_file_50embedtime1.pickle","wb") as tk:
                pickle.dump(self.tokenizer,tk,protocol=pickle.HIGHEST_PROTOCOL)

            self.total_words = len(self.tokenizer.word_index) + 1
            for each_line in lines:
                each_line = each_line.strip()
                self.token_list = self.tokenizer.texts_to_sequences([each_line])[0]
                for i in range(1,len(self.token_list)):
                    ngram_seq = self.token_list[:i+1]
                    self.encompass.append(ngram_seq)
        print(f" first stage {self.encompass} {self.total_words} {self.tokenizer}")
        return self.encompass,self.total_words,self.tokenizer
    
  
    
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
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
        return xs,ys,labels
    
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
        model_name_comp = f"{result_path}{model_name}"
        if tf.test.gpu_device_name():
            print(f"Default GPU device : {tf.test.gpu_device_name()}")
            with tf.device('/GPU:0'):
                loaded_model = load_model(model_name_comp,compile=True)
                # Reduce learning rate when a metric has stopped improving
                lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

                history = loaded_model.fit(xs,ys,epochs=50,verbose=1,callbacks=[lr_scheduler,early_stopping])

                file_name = f"{result_path}main_bilstm_scratch_model_150embedtime5.keras"
                if os.path.exists(file_name):
                    os.remove(file_name)

                loaded_model.save(file_name)

                with open(f"{result_path}main_historyrec_150embedtime5.pickle","wb") as hs:
                    pickle.dump(history,hs)
        else:
            print("Please install GPU version of TF")
            loaded_model = load_model(model_name_comp,compile=True)

            history = loaded_model.fit(xs,ys,epochs=50,verbose=1)

            file_name = f"{result_path}main_bilstm_scratch_model_150embedtime5.keras"
            if os.path.exists(file_name):
                os.remove(file_name)

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

    def consolidate_data_train(self,filepath,result_path):
        input_seq,total_words,tokenizer = self.tokenize_data_inp_seq(filepath,result_path)
        padd_seq,max_len = self.pad_sequ(input_seq)
        xs,ys,labels = self.prep_seq_labels(padd_seq,total_words)
        model_name = "main_bilstm_scratch_model_150embedtime4.keras"
        print(f"total words {total_words}")
        print(f"max len {max_len} xs {xs}")
        print(f"ys {ys}")
        print(f"result path {result_path}")
        history,model = self.train_model_five_runs(total_words,max_len,xs,ys,result_path)
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


    def evaluate_bilstm(self,test_data,maxlen,model_path,result_path):
        y_true = []
        i=0
        y_pred = []
        tokenz = None
        loaded_model = load_model(f"{result_path}{model_path}",compile=False)
        with open(f"{result_path}tokenized_file_50embedtime1.pickle","rb") as tk:
            tokenz = pickle.load(tk)
            
        
        # Start the evaluation timer
        start_time = time.time()

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            
            
            for line in lines:
               
                line = line.strip()
                
                sentence_tokens = line.split(" ")
            
                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]
                predicted_next_word = self.predict_token(context,tokenz,loaded_model,maxlen)
                
                
                i+=1
                if i%500 == 0:
                    
                    print(f"progress {i}")
            
                if predicted_next_word is not None:
                    y_true.append(true_next_word)
                
                    y_pred.append(predicted_next_word)
                

                print(f"trueword {true_next_word} context {context} predicted {predicted_next_word}")
                
                if len(y_true) == 0 or len(y_pred) == 0:
                    print("No valid predictions made.")
                    return None, None, None, None
        end_time = time.time()
        time_spent = end_time - start_time
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='weighted',zero_division=np.nan)
        f1score = f1_score(y_true,y_pred,average="weighted")

        with open(f"{result_path}bilstmmetrics_150embedtime1.txt","a") as blm:
            blm.write(f" another accuracy {accuracy} \n |  precision {precision} \n  |  recall {recall} \n  | f1score {f1score} \n  | evaluation time {time_spent:.2f} seconds \n")
        
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
        context = context.replace("_","UNDERSCORE")
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
        output_word  = output_word.replace("UNDERSCORE","_")
        return output_word

    def load_trained_model(self,model_name) :
        with open(model_name,"rb") as f:
            self.loaded_scratch_model = pickle.load(f)
        return self.loaded_scratch_model

    


    def train_model_five_runs(self, total_words, max_seq, xs, ys, result_path):
        print(tf.__version__)

        # Check for GPU availability
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

        # Define callbacks outside the loop to maintain consistency
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Run model training for 5 runs, reloading the model each time
        model_file_name = None
        for run in range(1, 6):
            print(f"\nStarting run {run}...\n")
            start_time = time.time()

            # Load the previous model if it exists, else create a new one
            if run == 1 or model_file_name is None:
                # First run or no saved model, initialize a new model
                model = Sequential([
                Embedding(total_words, 100, input_shape=(max_seq - 1,)),
                Bidirectional(LSTM(150)),
                Dense(total_words, activation='softmax')
                ])
                adam = Adam(learning_rate=0.01)
                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            else:
                # Load the model saved from the previous run
                run_curr = run - 1
                model_file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_{run_curr}.keras"
                model = load_model(model_file_name, compile=True)

            # Fit the model
            history = model.fit(xs, ys, epochs=50, verbose=1, callbacks=[lr_scheduler, early_stopping])

            # Save the history
            with open(f"{result_path}main_historyrec_150embedtime{run}.pickle", "wb") as hs:
                pickle.dump(history.history, hs)

            end_time = time.time()
            time_spent = end_time - start_time
            print(f"Run {run} complete. Training time: {time_spent:.2f} seconds")

            # Save the model and record training details
            model_file_name = f"{result_path}main_bilstm_scratch_model_150embedtime1_main_{run}.keras"
            model.save(model_file_name)

            with open(f"{result_path}main_seqlen_150embedtime{run}.txt", "a") as se:
                se.write(f"Run {run}: sequence length {max_seq}, training time {time_spent:.2f} seconds\n")

            print(f"Model for run {run} saved as {model_file_name}")
            

cl_ob = bi_lstm_scratch()
#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt")
#cl_ob.consolidate_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","bilstm_scratch_model_100embedtime2.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/models_gram/bi_lstm/results/results2/")

#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_00.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_portion/")

cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_150_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_150_v2/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_50_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_50/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_100_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_100/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_150_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_150/")
#cl_ob.consolidate_data_train("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_500_projects.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_500/")


#cl_ob.consolidate_data("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_test_data_10.txt","bilstm_scratch_model_50embedtime1.keras","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/bi_lstm/results_local/")
#cl_ob.plot_graph("loss")
#cl_ob.evaluate_bilstm("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_10_projects.txt",41,"main_bilstm_scratch_model_150embedtime1.keras","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/bilstm/models_10_v2/")
#cl_ob.predict_next_token_bilstm("event_whenflagclicked control_forever BodyBlock control_create_clone_of")