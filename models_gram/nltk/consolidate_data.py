import os
import pickle
import re
from itertools import product 

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


def write_each_train_file_opt(base_file_path, base_new_train_path):
    model_numbers = [10, 20, 30, 50, 80]
    ngram_range = range(2, 7)
    run_range = range(1, 6)

    # Generate all unique (each_number, ngram, run) combinations
    for each_number, ngram, run in product(model_numbers, ngram_range, run_range):
        each_file_path = f"{base_file_path}/{each_number}/path_{each_number}_{ngram}_{run}"
        new_file_name = f"{base_new_train_path}/scratch_train_set_{each_number}_{ngram}_{run}.pkl"

        # Skip writing if file already exists
        if os.path.exists(new_file_name):
            print(f"Skipping existing file: {new_file_name}")
            continue

        # Read file data only once per unique file path
        file_data = consolidate_data(each_file_path)

        # Write data to a unique file
        with open(new_file_name, 'wb') as file:
            pickle.dump(file_data, file)
            print(f"Written: {new_file_name}")


def load_data(filepath):
    with open(filepath,'rb') as fp:
        file = pickle.load(fp)
        print(type(file))
        #print(file)


def conv_pkl_to_txt_optimized(pickle_file_path,test_set_path):
    try:
        # Load the pickle file
        all_pickle_files = sorted([files for files in os.listdir(pickle_file_path) if files.endswith(".pkl")])
        for each_file in all_pickle_files:
            numbers = re.findall(r'\d+', each_file)
            if len(numbers) >= 3:
                model_number,ngram_number,run_number = map(int, numbers[:3])

                with open(f"{pickle_file_path}/{each_file}", 'rb') as pkl_file:
                    data_train_pkl = pickle.load(pkl_file)
                
                # Write the data to a text file
                with open(f"{test_set_path}/scratch_test_set_{model_number}_{ngram_number}_{run_number}.txt", 'w') as txt_file:
                    if isinstance(data_train_pkl, list):
                        # Write each element of the list on a new line
                        for ele in data_train_pkl:
                            txt_file.write(f"{ele}\n")
                    elif isinstance(data_train_pkl, dict):
                        # For dictionaries, print key-value pairs on new lines
                        for key, value in data_train_pkl.items():
                            txt_file.write(f"{key}: {value}\n")
                    else:
                        # Convert any other data type to string and write
                        txt_file.write(str(data_train_pkl))

    except (pickle.UnpicklingError, IOError) as e:
        print(f"Error processing the file: {e}")


def conv_pkl_to_txt_optimized2(pickle_file_path, test_set_path):
    try:
        # Get sorted list of all .pkl files
        all_pickle_files = sorted(f for f in os.listdir(pickle_file_path) if f.endswith(".pkl"))

        for each_file in all_pickle_files:
            numbers = re.findall(r'\d+', each_file)
            if len(numbers) < 3:
                continue  # Skip files that don't match expected pattern

            model_number, ngram_number, run_number = map(int, numbers[:3])

            pkl_file_path = os.path.join(pickle_file_path, each_file)
            txt_file_path = os.path.join(test_set_path, f"scratch_train_set_{model_number}_{ngram_number}_{run_number}.txt")

            with open(pkl_file_path, 'rb') as pkl_file:
                data_train_pkl = pickle.load(pkl_file)

            # Convert data to string format
            if isinstance(data_train_pkl, list):
                content = "\n".join(map(str, data_train_pkl))
            elif isinstance(data_train_pkl, dict):
                content = "\n".join(f"{key}: {value}" for key, value in data_train_pkl.items())
            else:
                content = str(data_train_pkl)

            # Write the data to a text file
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(content)

            print(f"Converted: {each_file} -> {txt_file_path}")

    except (pickle.UnpicklingError, IOError) as e:
        print(f"Error processing files: {e}")

def write_each_train_file_opt_corr(base_file_path, base_new_train_path):
    model_numbers = [10, 20, 30, 50, 80]
    ngram_range = range(2, 7)
    run_range = range(1, 6)

    for each_number, ngram, run in product(model_numbers, ngram_range, run_range):
        each_file_path = f"{base_file_path}/{each_number}/path_{each_number}_{ngram}_{run}"
        new_file_name = f"{base_new_train_path}/scratch_train_set_{each_number}_{ngram}_{run}.pkl"

        # Check if file exists
        if os.path.exists(new_file_name):
            try:
                with open(new_file_name, 'rb') as existing_file:
                    existing_data = pickle.load(existing_file)

                # Load new data
                file_data = consolidate_data(each_file_path)

                # Remove duplicates if file_data is a list (or similar structure)
                if isinstance(file_data, list):
                    file_data = list(set(file_data))  # Removing duplicate lines
                
                # Skip writing if the content is identical after removing duplicates
                if existing_data == file_data:
                    print(f"Skipping duplicate content: {new_file_name}")
                    continue
            except (pickle.UnpicklingError, EOFError):
                print(f"Corrupt file detected, overwriting: {new_file_name}")

        else:
            # Load new data only if needed
            file_data = consolidate_data(each_file_path)

            # Remove duplicates if file_data is a list
            if isinstance(file_data, list):
                file_data = list(set(file_data))  # Removing duplicate lines

        # Write the new data
        with open(new_file_name, 'wb') as file:
            pickle.dump(file_data, file)
            print(f"Written: {new_file_name}")

def write_each_train_file_opt_mn(base_file_path, base_new_train_path):
    model_numbers = [10, 20, 30, 50, 80]
    ngram_range = range(2, 7)
    run_range = range(1, 6)
    
    # Keep track of written content (using a set of hashes to check for duplicates)
    written_content = set()

    for each_number, ngram, run in product(model_numbers, ngram_range, run_range):
        each_file_path = f"{base_file_path}/{each_number}/path_{each_number}_{ngram}_{run}"
        new_file_name = f"{base_new_train_path}/scratch_train_set_{each_number}_{ngram}_{run}.pkl"

        # Check if file exists and skip if the file is already written
        if os.path.exists(new_file_name):
            try:
                with open(new_file_name, 'rb') as existing_file:
                    existing_data = pickle.load(existing_file)

                # If the content has already been written, skip
                data_hash = hash(str(existing_data))
                if data_hash in written_content:
                    print(f"Skipping duplicate content: {new_file_name}")
                    continue
                else:
                    written_content.add(data_hash)
            except (pickle.UnpicklingError, EOFError):
                print(f"Corrupt file detected, overwriting: {new_file_name}")

        # Load new data only if needed
        file_data = consolidate_data(each_file_path)

        # Check if the content has been written already, if yes, skip
        data_hash = hash(str(file_data))
        if data_hash in written_content:
            print(f"Skipping duplicate content: {new_file_name}")
            continue
        else:
            written_content.add(data_hash)

        # Write the new data
        with open(new_file_name, 'wb') as file:
            pickle.dump(file_data, file)
            print(f"Written: {new_file_name}")

def write_each_train_file_to_txt(base_file_path, base_new_train_path):
    model_numbers = [10, 20, 30, 50, 80]
    ngram_range = range(2, 7)
    run_range = range(1, 6)
    
    # Keep track of written content (using a set of hashes to check for duplicates)
    written_content = set()

    for each_number, ngram, run in product(model_numbers, ngram_range, run_range):
        each_file_path = f"{base_file_path}/{each_number}/path_{each_number}_{ngram}_{run}"
        new_file_name = f"{base_new_train_path}/scratch_train_set_{each_number}_{ngram}_{run}.txt"

        # Check if file already exists and skip if content is the same
        if os.path.exists(new_file_name):
            try:
                with open(new_file_name, 'r') as existing_file:
                    existing_data = existing_file.readlines()

                # If the content has already been written, skip
                data_hash = hash(str(existing_data))
                if data_hash in written_content:
                    print(f"Skipping duplicate content: {new_file_name}")
                    continue
                else:
                    written_content.add(data_hash)
            except IOError:
                print(f"Error reading file, overwriting: {new_file_name}")

        # Load new data only if needed
        file_data = consolidate_data(each_file_path)

        # Check if the content has been written already, if yes, skip
        data_hash = hash(str(file_data))
        if data_hash in written_content:
            print(f"Skipping duplicate content: {new_file_name}")
            continue
        else:
            written_content.add(data_hash)

        # Write the new data to a text file
        with open(new_file_name, 'w') as file:
            for line in file_data:
                file.write(f"{line}\n")  # Write each item as a new line
            print(f"Written: {new_file_name}")


#dump_data_in_pickle("scratch_data.pkl","/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/sb3_parsed/extracted_paths")
#dump_data_in_pickle("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/scratch_data_120_projects_model_test.pkl","/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram3/thesis_models/test_models/test_data/list_path_120_v2/")
#load_data("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_models_ngram/scratch_data_version3.pkl")
write_each_train_file_to_txt("/media/crouton/siwuchuk/newdir/vscode_repos_files/method","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/train_sets")
#conv_pkl_to_txt_optimized2("/media/crouton/siwuchuk/newdir/vscode_repos_files/method/train_sets","/media/crouton/siwuchuk/newdir/vscode_repos_files/method/datasets_train")