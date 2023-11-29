import json
import os
import hashlib

def calc_sha256(file_data):
    return hashlib.sha256(file_data.encode()).hexdigest() if isinstance(file_data, str) else hashlib.sha256(file_data).hexdigest()




def create_csv_hash_id(path):
    if os.path.isdir(path):
        for project_names in os.listdir(path):
            if not os.path.isdir(f'{path}/{project_names}'):
                continue
            else:
                if not os.listdir(f'{path}/{project_names}'):
                    continue
                else:
                    for filename in os.listdir(f'{path}/{project_names}'):
                        with open(os.path.join(f'{path}/{project_names}', filename), 'r') as f: # open in readonly mode
                            # read the json file
                            original_part = filename.split("_CMMT_")[0]
                            new_original_file_name = original_part.replace(",", "_COMMA_")
                            new_original_file_name = new_original_file_name.replace("_FFF_", "/")  
                            # suggestion: use the original file name extension and check if the filename is actually empty, in that case, no need to add extensions.    
                            new_original_file_name_sb3 = new_original_file_name + ".sb3" if ".sb3" not in new_original_file_name else new_original_file_name 
        
                            commit = filename.split("_CMMT_")[1].split(".json")[0]
                            print(commit)
                            data = json.load(f)
            
                            # convert data to string and assign to content column
                            content = str(data)
            
                            # calculate the hash value of the content
                            hash_value = calc_sha256(content)

                            # assign the folder name to the project_name column
                            project_name = project_names

                            # assign the filename to the file_name column
                            file_name = new_original_file_name_sb3

                            # assign the commit to the commit_sha column
                            commit_sha = commit

                        
            
                
                            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/hashcontents/project_file_commit_hash1.txt", "a") as outfile:
                                outfile.write("{},{},{},{}\n".format(project_name, file_name, commit_sha, hash_value))

create_csv_hash_id("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/projects")