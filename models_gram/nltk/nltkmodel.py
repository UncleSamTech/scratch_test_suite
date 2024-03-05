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
import subprocess
from nltk.lm import MLE
import re
from nltk.tokenize import ToktokTokenizer
#from nltk import word_tokenize,sent_tokenize

#from sklearn.model_selection import train_test_split

class nltkmodel:

    def __init__(self):
        self.ngrams = []
        self.token_text = None
        self.tokenized_text = None
        
    def escape_special_chars(self,input_string):
        escaped_string = subprocess.check_output(['echo', input_string], universal_newlines=True)
        escaped_string = subprocess.check_output(['sed', 's/[\\\/^$.*+?()[\]{}|[:space:]]/\\\\&/g'], input=escaped_string, universal_newlines=True)
        return escaped_string.strip()
    
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
    
    def tokenize(self,sent,n):
        try:
            from nltk import word_tokenize,sent_tokenize
            val = word_tokenize(sent_tokenize(sent.split()[0]))
            print(val)
        except:
            
            from nltk.tokenize import ToktokTokenizer
            sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])',x)
            toktok = ToktokTokenizer()
            word_tokenize = word_tokenize = toktok.tokenize
            self.tokenized_text = [list(map(str.lower,word_tokenize(sent_val))) for sent_val in sent_tokenize(self.escape_special_chars(sent))]
            train_data,padded_sents = padded_everygram_pipeline(n,self.tokenized_text)
            '''
            if os.path.isfile(file):
                with io.open(file,encoding='utf-8') as fin:
                    self.token_text = fin.read()
                    self.token_text = self.token_text.split()
                    self.tokenized_text = [list(map(str.lower,word_tokenize(sent_val))) for sent_val in sent_tokenize(self.token_text)]
                  '''  

        return train_data,padded_sents

    def train_ngrams(self,n,words):
        model = MLE(n)
        train_data,padded_sents = self.tokenize(words,n)
        model.fit(train_data,padded_sents)
        #print(model.vocab)
        #print(model.counts("event_whenflagclicked"))
        #print(model.score("event_whenflagclicked"))
        #print(model.vocab.lookup("event_whenflagclicked"))

    def large_train(self,file_name,n):
        if os.path.isfile(file_name):
            with open(file_name,"r") as fn:
                fn = fn.readlines()
                print(type(fn))
                for each_sent in fn:
                    print(type(each_sent))
                    #each_sent = each_sent.split()
                    print(each_sent)
                    #train = self.train_ngrams(n,each_sent)
                    print(train)

test_nltk = nltkmodel()
#v = test_nltk.create_ngrams(3,"files.txt","eslam-CS50.scratch event_whenflagclicked sensing_askandwait QUESTION What's your name?","<s>")
#w = test_nltk.tokenize("eslam-CS50.scratch event_whenflagclicked sensing_askandwait QUESTION What's your name?","files.txt")
#print("see",v)
train = test_nltk.train_ngrams(2,"eslam-CS50.scratch 2.3 - Monster Maze Untitled-2 copy-3 copy copy Angelo's Adventure T-Rex Watermelon Game cs50 - Martinho - globetroters Untitled-4 scratchcatgame Scratch Project Pac-Man Bat-Dodger airplane-scratch Untitled-8 (1) Converte_Real broadcast_special_chars comments_no_duplicate_id_serialization comments EZ-IXL Easy Scratch Project Moderate Scratch Project Mobinakashanian-final-doodle game-intro Star Wars - Rock Paper Scissors AP Snake - Eric Smith scratch-rocket-fly Kitty Cat Final- Problem Set 0's code via submit50! 02 - Projeto Scratch Scratch-Project-Assignment1 pfeilsteuerung toio-pe3roulette-20210506 Jim-Bob Bug Dinner - Game Scratch Project-2 Projekt Scratch-2 Find the Button Scratch - Snakes and Ladders Board asistente-ml-2 asistente-ml lab1task4-1 silly-couple-convo Scratch Project-5 Dribbling Ball v1 Ice-Hockey InvadersFromMars-5 ABC Furret OS 0.1-Alpha Cat-Tag Basic Python-Scratch Communication HUNGRY MONKEY-NEW sw1-1sb3 (Activity-4)C9186 Halit Untitled-3(1) Jogo da bola-cobrinha copy Untitled-8 Nefarious-Nebula Week0-Aharris5366 Scratch Project-8 Scratch Project-2-6 alien-language DismemberTheCat-v1 Untitled-2 (1) MUST-PUSH-BUTTON Demo With Storyline ECW1 - champion example creuza pongProject McCulloch Pitts Model XOR week 0 - scratch project bullet-reflect CS50-ProblemSet0 The Magic-less Fairy Dont_poke_the_bat fall report-repair Iron-man under-artist-app rock-paper-scissors yet another - Space Invaders Scratch Penalty Shootout Remix - adrianurdar disappearing-pen bouncy-ellipse-filler Problem 50-CS50x_Kiss the mutated toad Flappy Bird - Aakif moon-moods Spamux pre-alpha 0.1 IGRA_6-laba (1) Week 0-Project balloon-pop CW2 CH2-Amanda Mxi Bo-J_ScratchWithSunglasses AOCDay5 CS50 Problem 0 final- 'Fruit Splash' Untitled-60 Projeto Integrado - Cintia da Silva Costa C9147- Hilal- 4th Assignment (Premise) C9147- Hilal - 5th Activity(Characterization) SubojSWitherom-Template Legend of Zelda - Boss Battle Brick-Hill Unblocked Ver 1.4 (2) Clase 3 - Ejercicio 1 Un chocolatico uk-map-demo a Untitled-3 Project_ _GERONIMO!_ - Demo program3part1 Program3part2 event_whenflagclicked motion_pointindirection DIRECTION -90")
print(train)