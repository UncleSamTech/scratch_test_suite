#from nltk import ngrams
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
import io
import requests
import os
import re
from nltk.tokenize import ToktokTokenizer
from nltk import word_tokenize,sent_tokenize

#from sklearn.model_selection import train_test_split

class nltkmodel:

    def __init__(self):
        self.ngrams = []
        self.token_text = None
        self.tokenized_text = None
        

    
    def create_ngrams(self,n,file,sent,pad_symbol):
        '''
        with open(file,"r") as f:
            lines = f.readlines()
            for line in lines:
                print(line)
                print(type(line))
                if isinstance(line,str) and len(line) > 0:
                    self.ngrams.append(ngrams(line.split(),n))
                    '''
        sent = sent.split()
        #padded_sent = list(pad_sequence(sent,pad_left=True,left_pad_symbol=pad_symbol,pad_right=True,right_pad_symbol=pad_symbol,n=n))
        #print(list(bigrams(padded_sent)))
        padded_sent = pad_both_ends(sent,n=n)
        #everygram = list(everygrams(padded_sent,max_len=n))
       #flattened_gram = list(flatten(vals for vals in padded_sent))
        #train, vocab = padded_everygram_pipeline(n,padded_sent)
        training_ngrams, padded_sentences = padded_everygram_pipeline(n,padded_sent)
        for ngramilize_sent in training_ngrams:
            print(list(ngramilize_sent))
            print()
        print("#############")
        padd_sent = list(padded_sentences)
        print(padd_sent)
        #print(padd_sent)
        #print("flattened gram" , flattened_gram)
        #return list(ngrams(padded_sent,n))
        return padd_sent
    
    def tokenize(self,sent,file):
        try:
            from nltk import word_tokenize,sent_tokenize
            val = word_tokenize(sent_tokenize(sent.split()[0]))
            print(val)
        except:
            

            sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])',x)
            toktok = ToktokTokenizer()
            word_tokenize = word_tokenize = toktok.tokenize

            if os.path.isfile(file):
                with io.open(file,encoding='utf-8') as fin:
                    self.token_text = fin.read()
                    self.token_text = self.token_text.split()
                    self.tokenized_text = [list(map(str.lower,word_tokenize(sent_val))) for sent_val in sent_tokenize(self.token_text)]
                    

        return self.token_text


test_nltk = nltkmodel()
#v = test_nltk.create_ngrams(3,"files.txt","eslam-CS50.scratch event_whenflagclicked sensing_askandwait QUESTION What's your name?","<s>")
w = test_nltk.tokenize("eslam-CS50.scratch event_whenflagclicked sensing_askandwait QUESTION What's your name?","files.txt")
#print("see",v)
print("w",w)