import os
import pickle

data = []
def consolidate_data(connections_path):
    if os.path.isdir(connections_path):
        for scratch_connection_files in os.listdir(connections_path):
            with open(os.path.join(connections_path,scratch_connection_files),'r',encoding='utf-8') as scratch_connections:
                lines = scratch_connections.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) > 0:
                        data.append(line)
                    else:
                        continue
    return data

def dump_data_in_pickle(filename,file_path):
    data_file = consolidate_data(file_path)

    with open(filename,'wb') as file:
        pickle.dump(data_file,file)

def load_data(filepath):
    with open(filepath,'rb') as fp:
        file = pickle.load(fp)
        print(type(file))
        #print(file)

dump_data_in_pickle("scratch_data.pkl","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/sb3_parsed/extracted_paths")
#dump_data_in_pickle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/scratch_data_version4.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/list_of_hashes/extracted_paths5")
#load_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_data_version3.pkl")