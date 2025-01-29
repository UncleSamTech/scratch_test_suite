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

def write_each_train_file(base_file_path,base_new_train_path):
    model_numbers = [10,20,30,50,80]
    for each_number in model_numbers:
        for ngram in range(2,7):
            for run in range(1,6):
                each_file_path = f"{base_file_path}/{each_number}/path_{each_number}_{ngram}_{run}"
                file_data = consolidate_data(each_file_path)
                new_file_name = f"{base_new_train_path}/scratch_train_set_{each_number}_{ngram}_{run}.pkl"
                with open(new_file_name,'wb') as file:
                    pickle.dump(file_data,file)


def load_data(filepath):
    with open(filepath,'rb') as fp:
        file = pickle.load(fp)
        print(type(file))
        #print(file)

#dump_data_in_pickle("scratch_data.pkl","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/sb3_parsed/extracted_paths")
#dump_data_in_pickle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_120_projects_model_test.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/list_path_120_v2/")
#load_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_data_version3.pkl")
write_each_train_file("/media/crouton/siwuchuk/newdir/vscode_repos_files/method","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/train_sets")