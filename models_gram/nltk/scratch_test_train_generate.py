import os
import pickle
from sklearn.model_selection import train_test_split

def test_train_test_split(pickle_data,test_data_name,train_data_name):
    data_load = None
    with open(pickle_data,'rb') as file:
        data_load = pickle.load(file)

    train_data,test_data = train_test_split(data_load,test_size=0.1,random_state=42)

    with open(train_data_name,"w") as file_train:
        for token in train_data:
            file_train.write(token + '\n')
    
    with open(test_data_name,'w') as file_test:
        for token in test_data:
            file_test.write(token + '\n')


def conv_pkl_to_txt(pickle_file,train_output_file):
    data_train_pkl = None
    with open(pickle_file,'rb') as pkl_file:
        data_train_pkl = pickle.load(pkl_file)
        print(type(data_train_pkl))
    data_train = str(data_train_pkl)


    #with open(train_output_file,'w') as txt_file:
        #txt_file.write(data_train)


#test_train_test_split("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_data_version4.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_test_data_10.txt","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_train_data_90.txt")
#test_train_test_split("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_data.pkl","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_test_data_10.txt","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/models_gram/nltk/res_models/scratch_train_data_90.txt")
conv_pkl_to_txt("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_data_version_upd.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/models_train_test/scratch_train_data_90.txt")
    