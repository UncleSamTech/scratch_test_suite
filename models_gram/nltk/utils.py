import os
from nltk import word_tokenize

def gener_list_list(data):
        if not data:
            return [[]]
        res = [list(word_tokenize(line.strip())) for line in data if line.strip()]
        print(res)
        return res

def check_available_rank(list_tuples,true_word):
        rank = -1

        for ind,val in enumerate(list_tuples):
            if true_word.strip() == val[0].strip():
                rank = ind + 1
                print(rank)
                return rank
        
        return rank

check_available_rank([("looksunderscoreswitchbackdropto",0.50),("backdrop",0.25),("looksunderscoreswitchbackdropto",0.15),("leftangliteralrightang",0.10)],"backdrop")

gener_list_list(["eventunderscorewhenflagclicked", "eventunderscorewhenflagclicked looksunderscoreswitchbackdropto","eventunderscorewhenflagclicked looksunderscoreswitchbackdropto backdrop leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang leftangliteralrightang"])
