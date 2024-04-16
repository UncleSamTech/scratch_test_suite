import os
import pickle

class code_gpt_model:

    def __init__(self):
        self.model = None
        self.data=[]

    
    def convert_to_pickle(self,file_name):
        if os.path.isfile(file_name):
            with open(file_name,"r") as open_file:
                lines = open_file.readlines()
                for line in lines:
                    line = line.split()
                    self.data.append(line)
        
            with open("train.pkl","wb") as pkl:
                    pickle.dump(self.data,pkl)


cd_gpt = code_gpt_model()
cd_gpt.convert_to_pickle("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/codegpt/train.txt")

