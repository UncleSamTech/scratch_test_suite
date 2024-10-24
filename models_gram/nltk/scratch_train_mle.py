import os
import pickle
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
import nltk
import time
import matplotlib.pyplot as plt
#import pandas as pd
import random
import scipy.stats as stats
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_curve,f1_score

class scratch_train_mle:

    def __init__(self):
        self.scratch_model = None
        self.ngram_model = None
        self.loaded_scratch_model = None

    def train_mle(self,train_data,n,trained_model_data):

        with open(train_data,"r",encoding="utf-8") as f:
            lines = f.readlines()
            #sublines = lines[:10000]
            tokenized_scratch_data = [list(word_tokenize(sent.strip())) for sent in lines]
            train_data,padded_sents = padded_everygram_pipeline(n,tokenized_scratch_data)
        
        try:
            self.scratch_model = MLE(n)
            self.scratch_model.fit(train_data,padded_sents)

            with open(f'{trained_model_data}_{n}.pkl',"wb") as fd:
                pickle.dump(self.scratch_model,fd)
        except Exception as es:
            print("error as a result of ", es)

           
    def load_trained_model(self,model_name) :
        with open(model_name,"rb") as f:
            self.loaded_scratch_model = pickle.load(f)
            #print(type(self.loaded_scratch_model))
            #print(self.loaded_scratch_model.vocab)
            #print(self.loaded_scratch_model.counts("event_whenflagclicked"))
            #print(self.loaded_scratch_model.score("event_whenflagclicked"))
            #print(self.loaded_scratch_model.vocab.lookup("move"))
        return self.loaded_scratch_model
    
   

    def predict_next_scratch_token(self,model_name,context_data):
        loaded_model = self.load_trained_model(model_name)
        scratch_next_probaility_tokens = {}

        for prospect_token in loaded_model.vocab:
            print("see token" , prospect_token)
            #print(f"context data {context_data}")
            scratch_next_probaility_tokens[prospect_token] = loaded_model.score(prospect_token,context_data.split(" "))
        
        scratch_predicted_next_token = max(scratch_next_probaility_tokens,key=scratch_next_probaility_tokens.get)
        #print("predicted score ", scratch_next_probaility_tokens)
        return scratch_predicted_next_token
    
    def scratch_evaluate_model_nltk(self,test_data,model_name):

        y_true = []
        i=0
        y_pred = []

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            lines_lenght = len(lines)
            print("lenght",lines_lenght)
            offset_lenght = lines_lenght - 50
            new_lines = lines[:offset_lenght]
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()

                context = ' '.join(sentence_tokens[:-1])  # Use all words except the last one as context
                true_next_word = sentence_tokens[-1]

                predicted_next_word = self.predict_next_scratch_token(model_name,context)
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/seelogs_portion.txt","a") as fp:
                    fp.write(f"for context {context} next token {predicted_next_word}")
                    fp.write("\n")
                
                i+=1
                if i%500 == 0:
                    print("see it",i)
            
                y_true.append(true_next_word)
                y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1score = f1_score(y_true,y_pred,average="weighted")
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
    
    def scratch_evaluate_model_nltk_first(self,test_data,model_name):

        y_true = []
        i=0
        y_pred = []
        context = None
        true_next_word = None
        predicted_next_word = None

        with open(test_data,"r",encoding="utf-8") as f:
            lines= f.readlines()
            random.shuffle(lines)
            lines_lenght = len(lines)
            #print("lenght",lines_lenght)
            offset_lenght = lines_lenght - 50
            new_lines = lines[:offset_lenght]
            
            for line in lines:
                line = line.strip()
                sentence_tokens = line.split()
                
                if len(sentence_tokens) > 1:
                    print("first word ", sentence_tokens[1])
                    context = ' '.join(sentence_tokens[0:-1])  # Use all words except the first one as context
                    true_next_word = sentence_tokens[1]
                    #print("true next word ", true_next_word)
            
                    predicted_next_word = self.predict_next_scratch_token(model_name,context)
                    #print(f"compare {true_next_word} with predicted next word {predicted_next_word}")
                    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/seelogs.txt","a") as fp:
                        fp.write(f"for context {context} next token {predicted_next_word}")
                        fp.write("\n")
                
                    print(f"predicted {predicted_next_word} true word {true_next_word}")
                
                    i+=1
                    if i%500 == 0:
                        print("see it",i)
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)
                else:
                    context = ' '.join(sentence_tokens)  # Use all words except the first one as context
                    true_next_word = sentence_tokens[0]
                    predicted_next_word = self.predict_next_scratch_token(model_name,context)

                    i+=1
                    if i%500 == 0:
                        print("see it",i)
            
                    y_true.append(true_next_word)
                    y_pred.append(predicted_next_word)


        #self.plot_precision_recall_curve(y_true,y_pred,fig_name)
        accuracy = accuracy_score(y_true, y_pred,average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1score = f1_score(y_true,y_pred,average="weighted")
        #print(f"accuracy {accuracy} precisions {precision} recall {recall} f1score {f1score}")
        return accuracy,precision,recall,f1score
    
    def shuffle_test_data(self,input_string):
        if isinstance(input_string,str) and len(input_string) > 0:
            #convert to list
            list_string  = list(input_string)
            shuffled_list = random.shuffle(list_string)
            shuffled_res = ''.join(shuffled_list)
            return shuffled_res
    
    def convert_hours(self,seconds):
        return round((seconds / 3600),2)

    def plot_precision_recall_curve(self,plot_name):

        Accuracy = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367]
        Precision = [0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743,0.2264853658226743,0.22696845761011392]
        Recall = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367]
        F1 = [0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387,0.21718121691450387,0.2176643087019435]
        Ngrams = [2,3,4,5,6,7,8,9,10]

        Accuracy2 = [0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328,0.24589371980676328]
        Precision2 = [0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372,0.22721000350383372]
        Recall2 = [0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328,0.24589371980676328]
        F1_2 = [0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699,0.2179863698935699]
        Ngrams2 = [10,11,12,13,14,15]

        Accuracy3 = [0.025120772946859903,0.21690821256038648,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        Precision3  = [0.0033068915888476084,0.1932486508468289,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424]
        Recall3 = [0.025120772946859903,0.21690821256038648,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        F1_3 = [0.005844424726412303,0.1908012224104309,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736]
        Ngrams3 = [2,3,4,5,6,7,8,9]

        Accuracy4 = [0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        Precision4  = [0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424,0.19904575229610424]
        Recall4 = [0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222,0.2222222222222222]
        F1_4 = [0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736,0.19634627597060736]
        Ngrams4 = [10,11,12,13,14,15]
        

        Accuracy_plot2_8 = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407]
        Recall_plot2_8 = [0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407]
        Accuracy_plot9_15 = [0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328]
        Recall_plot9_15 = [0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328]
        Precision_plot2_8 = [0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743]
        Precision_plot9_15 = [0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372]
        f1_plot2_8 = [0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387]
        f1_plot9_15 = [0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699]
        

        ngram2_8 = list(range(2,9))
        ngram_9_15 = list(range(9,16))
        ngram_2_10 = list(range(2,11))
        ngram_11_19 = list(range(11,20))

        accuracy_2_10 = [0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        precision_2_10 = [0.00039637450273575164,0.001006502629642402,0.01410223779838277,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145]
        recall_2_10 = [0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        f1_2_10 = [0.0006595641494970354,0.0012696922764036857,0.01547662029383393,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823]

        accurracy_11_19 = [0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        precision_11_19 = [0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145]
        recall_11_19 = [0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037]
        f1_11_19 = [0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823]
        
        main_accuracy = [0.009372597103015442,0.026268210062553254,0.02728651883871236,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844]
        main_precision = [0.00250684777717733,0.02000157711620491,0.02054685262657442,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945,0.045356115385945]
        main_recall = [0.009372597103015442,0.026268210062553254,0.02728651883871236,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844,0.028803591097071844]
        main_f1 = [0.0033898931640232023,0.019506147406811364,0.020210073337690322,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567,0.022929111031605567]
        main_training_time = [self.convert_hours(177.9156),self.convert_hours(249.6270),self.convert_hours(349.6335),self.convert_hours(475.4877),self.convert_hours(642.8615),self.convert_hours(861.8450),self.convert_hours(1094.8200),self.convert_hours(1372.9215),self.convert_hours(1628.4691)]
        main_evaluation_time = [self.convert_hours(512.2936),self.convert_hours(1200.4897),self.convert_hours(2738.8720),self.convert_hours(5382.5739),self.convert_hours(9082.1960),self.convert_hours(13952.8228),self.convert_hours(20310.5438),self.convert_hours(24475.8971),self.convert_hours(34695.9437)]
        main_gram_range = list(range(2,11))

        plt.plot(main_gram_range, main_precision, label = "Precision")
        plt.plot(main_gram_range, main_recall, label = "Recall")
        plt.plot(main_gram_range,main_f1, label = "F1")
        plt.plot(main_gram_range, main_accuracy, label = "Accuracy")
        #plt.plot(main_gram_range,main_training_time,label="Training Time converted to hrs")
        #plt.plot(main_gram_range,main_evaluation_time,label="Evaluation Time converted to hrs")
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Nltk_Model Scores vs N-Gram Orders 2 - 10 on the portion')
        plt.legend()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}_2_10.pdf')
        #plt.show()

    def paired_t_test(self,nltk_2_10,nltk_11_19):
        if isinstance(nltk_2_10,list) and len(nltk_2_10) > 0 and isinstance(nltk_11_19,list) and len(nltk_11_19) > 0:
            test_val = stats.ttest_rel(nltk_2_10,nltk_11_19)
            #print(test_val)
            return test_val
        
    def wilcon_t_test(self,group1,group2):
        return stats.wilcoxon(group1,group2)
        
    def multiple_train(self,list_ngrams,test_data,model_name,train_data):
        final_result = {}
        for each_gram in list_ngrams:
            try:
                self.train_mle(train_data,each_gram,model_name)
                acc,precision,rec,f1_score = self.scratch_evaluate_model_nltk(test_data,f'{model_name}_{each_gram}.pkl')

                final_result[f'{each_gram}-gram_nltk'] = [acc,precision,rec,f1_score]
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/trained_data_prec_rec_acc.txt","a") as precs:
                    precs.write(f"{each_gram} order accuracy {acc} precision {precision} recall {rec} f1score {f1_score}")
                    precs.write("\n")
            except:
                final_result = {f'{each_gram}-gram_nltk':[0,0,0,0]}

        
        return final_result
    
    import time

    def multiple_train_time_metrics(self, list_ngrams, test_data, model_name, train_data):
        final_result = {}
        log_file = "/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/logs/trained_data_prec_rec_acc_portions.txt"
    
        for each_gram in list_ngrams:
            try:
                # Log the time for training the model
                train_start_time = time.time()  # Start time
                self.train_mle(train_data, each_gram, model_name)
                train_end_time = time.time()  # End time
                train_duration = train_end_time - train_start_time  # Calculate duration
            
                # Log the time for evaluating the model
                eval_start_time = time.time()  # Start time
                acc, precision, rec, f1_score = self.scratch_evaluate_model_nltk(test_data, f'{model_name}_{each_gram}.pkl')
                eval_end_time = time.time()  # End time
                eval_duration = eval_end_time - eval_start_time  # Calculate duration

                # Store results
                final_result[f'{each_gram}-gram_nltk'] = [acc, precision, rec, f1_score]

                #Log training and evaluation results, including time
                with open(log_file, "a") as precs:
                    precs.write(f"{each_gram}-gram order:\n")
                    precs.write(f"Training time: {train_duration:.4f} seconds\n")
                    precs.write(f"Evaluation time: {eval_duration:.4f} seconds\n")
                    precs.write(f"Accuracy: {acc}, Precision: {precision}, Recall: {rec}, F1 Score: {f1_score}\n")
                    precs.write("\n")
        
            except Exception as e:
                # In case of an exception, log the error and mark the result as 0
                final_result[f'{each_gram}-gram_nltk'] = [0, 0, 0, 0]
                with open(log_file, "a") as precs:
                    precs.write(f"Error training or evaluating {each_gram}-gram model: {e}\n")
    
        return final_result

    
    
tr_scr = scratch_train_mle()
#accuracy = tr_scr.paired_t_test([0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407],[0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328])
#print("accuracy parametric t-test result for nltk model ", accuracy)
#precision =tr_scr.paired_t_test([0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743],[0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372])
#print("precision parametric t-test for nltk model",precision)
#f1 = tr_scr.paired_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
#print("f1 parametric ttest for nltk model",f1)
#accuracy_wilcoxon = tr_scr.wilcon_t_test([0.003050640634533252,0.0036607687614399025,0.027455765710799267,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037],[0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037,0.0278625177954037])
#print("accuracy wilcoxon result for nltk model ", accuracy_wilcoxon)
#precision_wilcoxon =tr_scr.wilcon_t_test([0.00039637450273575164,0.001006502629642402,0.01410223779838277,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145],[0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145,0.03634422546717145])
#print("precision parametric wilcoxon for nltk model",precision_wilcoxon)
#f1_wilcoxon = tr_scr.wilcon_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
#print("f1 parametric ttest for nltk model",f1_wilcoxon)


#accuracy_wilcoxon_2 = tr_scr.wilcon_t_test([0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367,0.24541062801932367],[0.24589371980676328])
#print("accuracy wilcoxon result for nltk model 7 - 11 vs 12 - 16 ", accuracy_wilcoxon_2)
#precision_wilcoxon_2 =tr_scr.wilcon_t_test([0.22551918224779507,0.2264853658226743,0.2264853658226743,0.22696845761011392,0.22696845761011392],[0.22721000350383372])
#print("precision parametric t-test for nltk model 7 - 11 vs 12 - 16 ",precision_wilcoxon_2)
#f1_wilcoxon_2 = tr_scr.wilcon_t_test([0.0006595641494970354,0.0012696922764036857,0.01547662029383393,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823],[0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823,0.016052136283941823])
#print("f1 parametric wilcoxon test for nltk model ",f1_wilcoxon_2)
tr_scr.multiple_train_time_metrics([2,3,4,5,6,7,8,9,10],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_00.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_results/nltk/scratch_trained_model_nltk_portion","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_00.txt")
#tr_scr.train_mle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90.txt",8,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2")
#tr_scr.load_trained_model("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_7.pkl")
#tr_scr.scratch_evaluate_model_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_8.pkl") 
#tr_scr.plot_precision_recall_curve("nltk_evaluation_metrics_results_main_portion")
#tr_scr.scratch_evaluate_model_nltk()

# shuffle and split training and test set into four parts for training in parallel
#shuf /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80.txt | split -d -n r/4 - /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/train_models/train_data/scratch_train_data_80_ --additional-suffix=.txt
#shuf /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20.txt | split -d -n r/4 - /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/scratch_test_data_20_ --additional-suffix=.txt