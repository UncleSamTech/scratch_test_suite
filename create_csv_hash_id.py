import json
import os
import sys
import hashlib

def calc_sha256(file_data):
    return hashlib.sha256(file_data.encode()).hexdigest() if isinstance(file_data, str) else hashlib.sha256(file_data).hexdigest()

#folder= sys.argv[1]
#print(folder)
#folder_name = "/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_contents/"
folder_name = "/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/test/"


for filename in os.listdir(folder_name):
    
    with open(os.path.join(folder_name, filename), 'r') as f: # open in readonly mode
        # read the json file
        original_part = filename.split("_CMMT_")[0]
        new_original_file_name = original_part.replace(",", "_COMMA_")
        new_original_file_name = new_original_file_name.replace("_FFF_", "/")  
        # suggestion: use the original file name extension and check if the filename is actually empty, in that case, no need to add extensions.    
        new_original_file_name_pd = new_original_file_name + ".sb3" if ".sb3" not in new_original_file_name else new_original_file_name 
        
        commit = filename.split("_CMMT_")[1].split(".json")[0]
        print(commit)
        if os.stat(os.path.join(folder_name, filename)).st_size != 0:
            
            data = json.load(f)
            
            # convert data to string and assign to content column
            content = str(data)
            
            # calculate the hash value of the content
            hash_value = calc_sha256(content)

            # assign the folder name to the project_name column
            project_name = folder

            # assign the filename to the file_name column
            file_name = new_original_file_name_pd

            # assign the commit to the commit_sha column
            commit_sha = commit

            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/hashcontents/project_file_commit_hash1.txt", "a") as outfile:
                outfile.write("{},{},{},{}\n".format(project_name, file_name, commit_sha, hash_value))

