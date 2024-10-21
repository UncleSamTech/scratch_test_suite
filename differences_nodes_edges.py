import sys
import json
from pydriller.git import Git
import os
from datetime import datetime
import subprocess
from pathlib import Path
import sqlite3

#conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db')

def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_all_projects_in_db():
    select_projects = """SELECT Project_Name from revisions;"""
    val = []

    
    fin_resp = []
    conn,curr = get_connection()
    if conn != None:
         curr.execute(select_projects)  
         val = curr.fetchall()
         fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
                     
    else:
        print("connection failed")
    conn.commit()
    return fin_resp

# def get_all_projects_in_db_optimized():
#     select_projects = """SELECT Project_Name FROM revisions;"""
#     fin_resp = []

#     # Establish the database connection
#     conn, curr = get_connection()

#     if conn is None:
#         print("Connection failed")
#         return fin_resp  # Return empty list on failure

#     try:
#         # Execute the query and fetch all results
#         curr.execute(select_projects)
#         val = curr.fetchall()

#         # Flatten the result list if it contains tuples
#         fin_resp = [each_val[0] for each_val in val if isinstance(each_val, tuple) and each_val]

#     except Exception as e:
#         print(f"Error fetching projects: {e}")

#     finally:
#         # Ensure the connection is closed
#         conn.commit()
#         conn.close()

#     return fin_resp

def get_content_parents_of_c(project_name, file_name, c):
    conn,curr = get_connection()
    if conn is None:
        print("Connection failed")
        return None  

    try:
        curr.execute("SELECT Content_Parent_SHA FROM Content_Parents WHERE Project_Name = ? AND File = ? AND Commit_SHA = ?", (project_name, file_name, c))
        all_parents_of_c = curr.fetchall()
        all_parents_of_c = set([x[0] for x in all_parents_of_c])
        return all_parents_of_c
    except Exception as e:
        print(f"Error fetching content parents: {e}")
    finally:
        # Ensure the connection is closed
        conn.commit()
        conn.close()
    

def get_node_and_edge_count(project_name, file_name, c):
    project_name = project_name.strip()
    file_name = file_name.strip()
    c = c.strip()

    conn,curr = get_connection()
    if conn is None:
        print("Connection failed")

    try:
        curr.execute("SELECT Nodes, Edges FROM Revisions WHERE Project_Name = ? AND File = ? AND Commit_SHA = ?", (project_name, file_name, c))
        nodes_edges = curr.fetchall()
        curr.close()
        #print(f"nodes edges list for {project_name} and {file_name} is {nodes_edges}")
        node_count = nodes_edges[0][0]
        edge_count = nodes_edges[0][1]
        
        #print(f"all project {project_name} filename {file_name} {nodes_edges} nodes {node_count} edges {edge_count}")
        return node_count, edge_count
    except Exception as e:
        print(f"error fetching nodes ane edges  for {project_name} and {file_name} {e}")
        return 0,0
    finally:
        conn.commit()
        conn.close()
    
    

# def get_node_and_edge_count_optimized(project_name, file_name, commit_sha):
#     node_count, edge_count = 0, 0  # Default values
#     project_name = project_name.strip()
#     file_name = file_name.strip()
#     commit_sha = commit_sha.strip()
#     try:
#         # Open a cursor using context management (ensures automatic cleanup)
#         with conn.cursor() as cursor:
#             cursor.execute(
#                 """SELECT Nodes, Edges 
#                    FROM Revisions 
#                    WHERE Project_Name = ? AND File = ? AND Commit_SHA = ?""",
#                 (project_name, file_name, commit_sha)
#             )

#             # Fetch the result and check if it exists
#             nodes_edges = cursor.fetchone()
            
#             if nodes_edges:
#                 node_count, edge_count = nodes_edges[0], nodes_edges[1]
#                 print(f"nodes: {node_count}, edges: {edge_count}")
#             else:
#                 print(f"No data found for project {project_name}, file {file_name}, commit {commit_sha}")

#     except Exception as e:
#         print(f"Error fetching node and edge count: {e}")

#     return node_count, edge_count


def get_revisions_and_run_parser(cwd, project_name, main_branch, debug=False):
    proc1 = subprocess.run(['git --no-pager log --pretty=tformat:"%H" {} --no-merges'.format(main_branch)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
    proc2 = subprocess.run(['xargs -I{} git ls-tree -r --name-only {}'], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc3 = subprocess.run(['grep -i "\\.sb3$"'], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc4 = subprocess.run(['sort -u'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    filenames = proc4.stdout.decode().strip().split('\n')

  
    if len(filenames) > 1:
        # for all pd files in ths project
        for f in filenames:
            proc1 = subprocess.run(['git --no-pager log -z --numstat --follow --pretty=tformat:"{}¬%H" -- "{}"'.format(f,f)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
            
            proc2 = subprocess.run(["cut -f3"], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            
            
            proc3 = correct_code_replace(proc2.stdout)
            proc4 = subprocess.run(['xargs -0 echo'], input=proc3, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            
            #decoded_proc4 = proc4.stdout.decode()
            
            filename_shas = proc4.stdout.decode().strip().split('\n')
            
            filename_shas = [x for x in filename_shas if x != '']

            #if 2 ¬ then it is the original filename that we are trying to trace back (includes original filename, commit)
            #if 3 ¬ then it is not renamed (includes renamed filename, original filename, commit)
            #if starts with ¬ then it is renamed (includes pre-filename, renamed filename, original filename, commit)
            #if 1 ¬ then it is a beginning file with no diff; skip
            # git log --all : commit history across all branches
            proc1 = subprocess.run(['git --no-pager log --all --pretty=tformat:"%H" -- "{}"'.format(f)], stdout=subprocess.PIPE, cwd=cwd, shell=True) # Does not produce renames
            all_shas = proc1.stdout.decode().strip().split('\n') 
            all_shas = [x for x in all_shas if x != '']
            all_sha_names = {}
        
            for x in all_shas:
                all_sha_names[x] = None

            # get filenames for each commit
            for fn in filename_shas: # start reversed, oldest to newest
                separator_count = fn.strip().count('-')
                split_line = fn.strip('-').split('-')
                
            
                if separator_count == 2:
                    c = split_line[-1]

                    if not is_sha1(c):
                        # Edge case where line doesn't have a sha
                        #print(split_line)
                        continue

                    all_sha_names[c] = split_line[0]
                        #print("Separator count 2: assigning {} to {}".format(c, split_line[0]))
        
                elif fn[0] == '-':
                    new_name = split_line[0]
                    c = split_line[-1]

                    if not is_sha1(c):
                        # Edge case where line doesn't have a sha
                        #print(split_line)
                        continue

                    all_sha_names[c] = new_name
                    #print("starting with separator: assigning {} to {}".format(c, split_line[-4]))
                
                elif separator_count == 3:
                    # print(split_line[-1])
                    new_name = split_line[0]
                    c = split_line[-1]

                    if not is_sha1(c):
                        # Edge case where line doesn't have a sha
                        #print(split_line)
                        continue

                    all_sha_names[c] = new_name
                    #print("Separator count 3: assigning {} to {}".format(c, split_line[-3]))
                
                elif separator_count == 1:
                    continue
        
                else:
                    raise ValueError('Unknown case for file')
 
            # fill in the gaps
            prev_fn = f
            for c in all_sha_names.keys():
                if all_sha_names[c] is None:
                    all_sha_names[c] = prev_fn
                prev_fn = all_sha_names[c]
        

            commits_which_modified_file_f = set(all_sha_names.keys()) # commits across all branches which modified file f
        
            for c in commits_which_modified_file_f:

                diff_node_count = 0
                diff_edge_count = 0
                
                file_name = f.replace(",", "_COMMA_")
                node_count_of_f_at_c, edge_count_of_f_at_c = get_node_and_edge_count(project_name, file_name, c)
                print(f"node count {node_count_of_f_at_c}, edge count {edge_count_of_f_at_c}")
                content_parents_of_c = get_content_parents_of_c(project_name, file_name, c)

                for parent in content_parents_of_c:
                    if parent == c:
                        diff_node_count = node_count_of_f_at_c
                        diff_edge_count = edge_count_of_f_at_c
                    else:
                        node_count_of_f_at_parent, edge_count_of_f_at_parent = get_node_and_edge_count(project_name, file_name, parent)
                        diff_node_count += (node_count_of_f_at_c - node_count_of_f_at_parent)
                        diff_edge_count += (edge_count_of_f_at_c - edge_count_of_f_at_parent)

                print(f"project_name {project_name} filename {file_name} commit {c} diff node count {diff_node_count} diff edge count {diff_edge_count}")
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/differences_nodes_edges/differences_nodes_edges_sb3_files.csv", "a") as outfile:
                    outfile.write("{},{},{},{},{}\n".format(project_name, file_name, c, str(diff_node_count), str(diff_edge_count)))

                    

        return 1
    
# def get_revisions_and_run_parser_optimized(cwd, project_name, main_branch, debug=False):
#     try:
#         # Get all commit hashes from the main branch excluding merge commits
#         proc1 = subprocess.run(
#             ['git --no-pager log --pretty=tformat:"%H" {} --no-merges'.format(main_branch)],
#             stdout=subprocess.PIPE, cwd=cwd, shell=True
#         )
        
#         # List files in each commit, filter for `.sb3` files, and get unique filenames
#         proc2 = subprocess.run(
#             ['xargs -I{} git ls-tree -r --name-only {}'],
#             input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True
#         )
#         proc3 = subprocess.run(
#             ['grep -i "\.sb3$"'], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True
#         )
#         proc4 = subprocess.run(
#             ['sort -u'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True
#         )
        
#         # Decode filenames and filter out any empty strings
#         filenames = proc4.stdout.decode().strip().split('\n')
#         filenames = [f for f in filenames if f]

#         if not filenames:
#             print(f"No .sb3 files found in project {project_name}.")
#             return

#         # Process each filename
#         for f in filenames:
#             # Get the full commit history for the file
#             proc1 = subprocess.run(
#                 ['git --no-pager log -z --numstat --follow --pretty=tformat:"{}¬%H" -- "{}"'.format(f, f)],
#                 stdout=subprocess.PIPE, cwd=cwd, shell=True
#             )
#             proc2 = subprocess.run(
#                 ["cut -f3"], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True
#             )

#             # Process the output for filename SHAs
#             proc3 = correct_code_replace(proc2.stdout)
#             proc4 = subprocess.run(['xargs -0 echo'], input=proc3, stdout=subprocess.PIPE, cwd=cwd, shell=True)

#             # Parse and clean the SHA data
#             filename_shas = proc4.stdout.decode().strip().split('\n')
#             filename_shas = [x for x in filename_shas if x]

#             # Map all SHAs to filenames (initialize None)
#             proc1 = subprocess.run(
#                 ['git --no-pager log --all --pretty=tformat:"%H" -- "{}"'.format(f)], stdout=subprocess.PIPE, cwd=cwd, shell=True
#             )
#             all_shas = proc1.stdout.decode().strip().split('\n')
#             all_sha_names = {x: None for x in all_shas if x}

#             # Assign filenames to SHAs based on their separation pattern
#             for fn in filename_shas:
#                 separator_count = fn.strip().count('-')
#                 split_line = fn.strip('-').split('-')

#                 if separator_count == 2 and is_sha1(split_line[-1]):
#                     all_sha_names[split_line[-1]] = split_line[0]
#                 elif fn[0] == '-' and is_sha1(split_line[-1]):
#                     all_sha_names[split_line[-1]] = split_line[0]
#                 elif separator_count == 3 and is_sha1(split_line[-1]):
#                     all_sha_names[split_line[-1]] = split_line[0]
#                 elif separator_count == 1:
#                     continue
#                 else:
#                     raise ValueError('Unknown case for file')

#             # Fill in gaps in the SHA-filename mapping
#             prev_fn = f
#             for c in all_sha_names.keys():
#                 if all_sha_names[c] is None:
#                     all_sha_names[c] = prev_fn
#                 prev_fn = all_sha_names[c]

#             commits_which_modified_file_f = set(all_sha_names.keys())

#             # Calculate node and edge differences for each commit
#             for c in commits_which_modified_file_f:
#                 node_count_of_f_at_c, edge_count_of_f_at_c = get_node_and_edge_count_optimized(project_name, f, c)
#                 content_parents_of_c = get_content_parents_of_c(project_name, f, c)

#                 diff_node_count, diff_edge_count = 0, 0
#                 for parent in content_parents_of_c:
#                     if parent == c:
#                         diff_node_count = node_count_of_f_at_c
#                         diff_edge_count = edge_count_of_f_at_c
#                     else:
#                         node_count_of_f_at_parent, edge_count_of_f_at_parent = get_node_and_edge_count_optimized(
#                             project_name, f, parent)
#                         diff_node_count += (node_count_of_f_at_c - node_count_of_f_at_parent)
#                         diff_edge_count += (edge_count_of_f_at_c - edge_count_of_f_at_parent)

#                     # Save results to the output file
#                     with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/differences_nodes_edges/differences_final_new_update_new_optimized_upd_3.csv", "a") as outfile:
#                         outfile.write("{},{},{},{},{}\n".format(project_name, f, c, str(diff_node_count), str(diff_edge_count)))

#         return 1

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return 0
                
def is_valid_encoding(byte_string,encoding='utf-8'):
    try:
        decoded_string = byte_string.decode(encoding)
        print(type(decoded_string))
        return True
    except UnicodeDecodeError:
        print(type(decoded_string))
        return False


def correct_code_replace(byte_val):
    if (is_valid_encoding(byte_val)):
        #decode the byte
        
        
        decoded_byte = byte_val.decode('utf-8')
        
        
        #replace values
        replace_val = decoded_byte.replace('\xc2','¬') if '\xc2' in decoded_byte else decoded_byte
        replace_val = replace_val.replace('\x00','¬') if '\x00' in replace_val else replace_val
        
        
        #replace_val = decoded_byte.replace('\xc2','#').replace('\x00','#') if '\xc2' in decoded_byte or '\x00' in decoded_byte else decoded_byte
        
        #encode it back
        encoded_byte = replace_val.encode('utf-8')
        
        #print(f"original {byte_val} after decode {decoded_byte} after replacement {replace_val} final value {encoded_byte}")
      
        return encoded_byte


def main2_optimized(project_path: str):
    #proj_names = [i for i in os.listdir(project_path) if os.path.isdir(f'{project_path}/{i}') and len(i) > 1]
    proj_names = get_all_projects_in_db()

    for proj_name in proj_names:
        repo = f'{project_path}/{proj_name}'
        try:
            # Get the current branch name
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=False)
            main_branch = result.stdout.decode("utf-8").strip()
            print(f"main branch {main_branch}")
            # Check if main_branch and repo are valid
            if main_branch and repo:
                # Run the parser if the branch name is valid
                get_revisions_and_run_parser(repo, proj_name, main_branch)
            else:
                print(f"Skipped project: {proj_name}")

        except Exception as e:
            # Use 'with' to ensure file is closed properly
            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/differences_nodes_edges/differences_nodes_edges_exceptions_main.txt", "a") as f:
                f.write(f"Error with project {proj_name}: {e}\n")


def main2(project_path: str):
    proj_names = []
    for i in os.listdir(project_path):
        if len(i) > 1 and os.path.isdir(f'{project_path}/{i}'):
            print(project_path)
            proj_names.append(i)
        else:
            continue
    
    #all_projects = get_all_projects_in_db()
   
    for proj_name in proj_names:
        if proj_name != '' and len(proj_name) > 1:
            repo = f'{project_path}/{proj_name}'
            main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
            main_branch = main_branch.stdout.decode("utf-8").strip('/n')[0:]
            
            if len(main_branch) > 1 or main_branch != '' or main_branch != None and repo != '' or repo != None and len(repo) > 0 and len(main_branch) > 0:
                try:
                    get_revisions_and_run_parser(repo, proj_name, main_branch)

                    
                except Exception as e:
                    
                    f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/differences_nodes_edges/differences_nodes_edges_exceptions3.txt", "a")
                    f.write("{}\n".format(e))
                    f.close()
                    #logging.error(f'skipped {project_name}  to {logging.ERROR}')
                #connection.commit()
                #connection.close()
                
                #connection.close()
            else:
                print("skipped")
                continue
        else:
            print("skipped")
            continue
    

#main2("/mnt/c/Users/USER/Documents/scratch_tester/scratch_test_suite/files/repos")
main2_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")