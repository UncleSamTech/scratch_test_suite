import os
import json

def get_nodes_edges_per_file(path):
    
    if os.path.isdir(path):
        for project_names in os.listdir(path):
            nodes_in_this_project = 0 
            edges_in_this_project = 0 
            if not os.path.isdir(f'{path}/{project_names}'):
                continue
            else:
                if not os.listdir(f'{path}/{project_names}'):
                    continue
                else:
                    for filename in os.listdir(f'{path}/{project_names}'):
                        #print(filename)
                        with open(os.path.join(f'{path}/{project_names}', filename), 'r') as f: # open in readonly mode
                            # read the json file
                            
                            original_part = filename.split("_CMMT_")[0]
                            
                            new_original_file_name = original_part.replace(",", "_COMMA_")
                            
                            new_original_file_name = new_original_file_name.replace("_FFF_", "/") 
                            
                            # suggestion: use the original file name extension and check if the filename is actually empty, in that case, no need to add extensions.    
                            new_original_file_name_sb3 = new_original_file_name + ".sb3" if ".sb3" not in new_original_file_name else new_original_file_name 
                            
                            commit = filename.split("_CMMT_")[1].split(".json")[0]
                            
                            data = json.load(f)
                            
                            an_file = new_original_file_name_sb3.split("/")[-1] if "/" in new_original_file_name_sb3 else new_original_file_name_sb3
                            an_file = os.path.splitext(an_file)[0] if ".sb3" in an_file else an_file
                            an_file = f'{an_file}_summary'
                            
                            try:
                                nodes_in_sb3_file  = data["stats"][an_file]["number_of_nodes"]  
                                edges_in_sb3_file = data["stats"][an_file]["number_of_edges"]
                            except:
                                nodes_in_sb3_file = 0
                                edges_in_sb3_file = 0
                            
                            
                            #with open("/mnt/c/Users/USER/Documents/scratch_tester/scratch_test_suite/files/sb3_parsed/nodes_edges/nodes_edges_per_file.csv", 'a') as f:
                            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/nodes_edges_per_file2.csv", 'a') as f:
                                f.write(project_names  + "," + new_original_file_name_sb3 + "," + commit + "," + str(nodes_in_sb3_file) + "," + str(edges_in_sb3_file))
                                f.write("\n")
                            
                            nodes_in_this_project += nodes_in_sb3_file
                            edges_in_this_project += edges_in_sb3_file
            
            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/nodes_edges/nodes_edges_folder/nodes_edges_per_project2.csv", 'a') as f:
            #with open("/mnt/c/Users/USER/Documents/scratch_tester/scratch_test_suite/files/sb3_parsed/nodes_edges/nodes_edges_per_project.csv", 'a') as f:
                f.write(project_names + "," + str(nodes_in_this_project) + "," + str(edges_in_this_project))
                f.write("\n")



                            
#get_nodes_edges_per_file("/mnt/c/Users/USER/Documents/scratch_tester/scratch_test_suite/files/sb3_parsed/new_projects")
get_nodes_edges_per_file("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/project2")

                        