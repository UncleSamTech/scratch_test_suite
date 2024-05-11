import os
import pickle
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
import nltk
import matplotlib.pyplot as plt
#import pandas as pd
import random
import scipy.stats as stats
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
                with open("seelogs.txt","a") as fp:
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
                    context = ' '.join(sentence_tokens[1:])  # Use all words except the first one as context
                    true_next_word = sentence_tokens[0]
                    #print("true next word ", true_next_word)
            
                    predicted_next_word = self.predict_next_scratch_token(model_name,context)
                    #print(f"compare {true_next_word} with predicted next word {predicted_next_word}")
                    with open("seelogs.txt","a") as fp:
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
        
        plt.plot(Ngrams3, Accuracy3, label = "Accuracy",color="blue",marker="o",linestyle="-")
        plt.plot(Ngrams3, Precision3, label = "Precision",color="red")
        plt.plot(Ngrams3, Recall3, label = "Recall",color="yellow")
        plt.plot(Ngrams3, F1_3, label = "F1",color="green")
        
        plt.xlabel('Ngram-order')
        plt.ylabel('Model-Scores')
        plt.title('Nltk_Model Scores vs N-Gram Orders for replaced tokens')
        plt.legend()
        #plt.xlim(min(Ngrams3), max(Ngrams3))
        #plt.ylim(min(min(Accuracy3), min(Precision3), min(Recall3), min(F1_3)), max(max(Accuracy3), max(Precision3), max(Recall3), max(F1_3)))

        plt.savefig(f'{plot_name}.pdf')
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
                acc,precision,rec,f1_score = self.scratch_evaluate_model_nltk_first(test_data,f'{model_name}_{each_gram}.pkl',each_gram)

                final_result[f'{each_gram}-gram_nltk'] = [acc,precision,rec,f1_score]
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/trained_data_prec_rec_acc_first.txt","a") as precs:
                    precs.write(f"{each_gram} order accuracy {acc} precision {precision} recall {rec} f1score {f1_score}")
                    precs.write("\n")
            except:
                final_result = {f'{each_gram}-gram_nltk':[0,0,0,0]}

        
        return final_result

    
    
tr_scr = scratch_train_mle()
#accuracy = tr_scr.paired_t_test([0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407],[0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328])
#print("accuracy parametric t-test result for nltk model ", accuracy)
#precision =tr_scr.paired_t_test([0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743],[0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372])
#print("precision parametric t-test for nltk model",precision)
#f1 = tr_scr.paired_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
#print("f1 parametric ttest for nltk model",f1)
accuracy_wilcoxon = tr_scr.wilcon_t_test([0.025120772946859903,0.2314009661835749,0.23719806763285023,0.2400966183574879,0.2429951690821256,0.24396135265700483,0.24492753623188407],[0.24492753623188407,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24541062801932367,0.24589371980676328])
print("accuracy wilcoxon result for nltk model ", accuracy_wilcoxon)
precision_wilcoxon =tr_scr.wilcon_t_test([0.0033068915888476084,0.20619551075021053,0.2124757039869255,0.22165444794827815,0.22455299867291584,0.22551918224779507,0.2264853658226743],[0.2264853658226743,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22696845761011392,0.22721000350383372])
print("precision parametric t-test for nltk model",precision_wilcoxon)
f1_wilcoxon = tr_scr.wilcon_t_test([0.005844424726412303,0.2026847567047111,0.20871290205232712,0.21235029904010772,0.21524884976474543,0.21621503333962466,0.21718121691450387],[0.21718121691450387,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2176643087019435,0.2179863698935699])
print("f1 parametric ttest for nltk model",f1_wilcoxon)


#accuracy_wilcoxon_2 = tr_scr.wilcon_t_test([0.24396135265700483,0.24492753623188407,0.24492753623188407,0.24541062801932367,0.24541062801932367],[0.24589371980676328])
#print("accuracy wilcoxon result for nltk model 7 - 11 vs 12 - 16 ", accuracy_wilcoxon_2)
#precision_wilcoxon_2 =tr_scr.wilcon_t_test([0.22551918224779507,0.2264853658226743,0.2264853658226743,0.22696845761011392,0.22696845761011392],[0.22721000350383372])
#print("precision parametric t-test for nltk model 7 - 11 vs 12 - 16 ",precision_wilcoxon_2)
#f1_wilcoxon_2 = tr_scr.wilcon_t_test([0.21621503333962466,0.21718121691450387,0.21718121691450387,0.2176643087019435,0.2176643087019435],[,0.2179863698935699])
#print("f1 parametric ttest for nltk model 7 - 11 vs 12 - 16 ",f1_wilcoxon_2)
#tr_scr.multiple_train([2,3,4,5,6,7,8,9,10,11,12,13,14,15],"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_trained_model_version4","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt")
#tr_scr.train_mle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_train_data_90.txt",8,"/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2")
#tr_scr.load_trained_model("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_7.pkl")
#tr_scr.scratch_evaluate_model_nltk("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_trained_model_version2_8.pkl") 
#tr_scr.plot_precision_recall_curve("nltk-plot_firstmain_replaced_tokens_upd_debug_mark_line_further")