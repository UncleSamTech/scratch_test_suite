import sys
import json
from pydriller.git import Git
import os
from datetime import datetime
import subprocess
from pathlib import Path
import sqlite3

conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database.db')

def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database.db",isolation_level=None)
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


def get_content_parents_of_c(project_name, file_name, c):
    
    cursor = conn.cursor()
    cursor.execute("SELECT Content_Parent_SHA FROM Content_Parents WHERE Project_Name = ? AND File = ? AND Commit_SHA = ?", (project_name, file_name, c))
    all_parents_of_c = cursor.fetchall()
    cursor.close()
    all_parents_of_c = set([x[0] for x in all_parents_of_c])
    return all_parents_of_c

def get_node_and_edge_count(project_name, file_name, c):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT Nodes, Edges FROM Revisions WHERE Project_Name = ? AND File = ? AND Commit_SHA = ?", (project_name, file_name, c))
        nodes_edges = cursor.fetchall()
        cursor.close()
        node_count = nodes_edges[0][0]
        edge_count = nodes_edges[0][1]
    except:
        node_count = 0
        edge_count = 0
    
    return node_count, edge_count


def get_revisions_and_run_parser(cwd, project_name, main_branch, debug=False):
    proc1 = subprocess.run(['git --no-pager log --pretty=tformat:"%H" {} --no-merges'.format(main_branch)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
    proc2 = subprocess.run(['xargs -I{} git ls-tree -r --name-only {}'], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    proc3 = subprocess.run(['grep -i "\.sb3$"'], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    proc4 = subprocess.run(['sort -u'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    filenames = proc4.stdout.decode().strip().split('\n')

    if filenames is None or filenames  == [''] or len(filenames) == 0 or filenames == []:
        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
        return -1
    else:
        # for all pd files in ths project
        for f in filenames:
            proc1 = subprocess.run(['git --no-pager log -z --numstat --follow --pretty=tformat:"{}¬%H" -- "{}"'.format(f,f)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc2 = subprocess.run(["cut -f3"], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc3 = subprocess.run(["sed 's/\d0/-/g'"], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc4 = subprocess.run(['xargs -0 echo'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
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
                #print(split_line)
                file_contents = ''
            
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
                content_parents_of_c = get_content_parents_of_c(project_name, file_name, c)

                for parent in content_parents_of_c:
                    if parent == c:
                        diff_node_count = node_count_of_f_at_c
                        diff_edge_count = edge_count_of_f_at_c
                    else:
                        node_count_of_f_at_parent, edge_count_of_f_at_parent = get_node_and_edge_count(project_name, file_name, parent)
                        diff_node_count += (node_count_of_f_at_c - node_count_of_f_at_parent)
                        diff_edge_count += (edge_count_of_f_at_c - edge_count_of_f_at_parent)
                
                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results/differences_final_new_update.csv", "a") as outfile:
                    outfile.write("{}_COMMA_{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, c, str(diff_node_count), str(diff_edge_count)))

                    

        return 1
                


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
                    print(repo)
                    print(proj_name)
                    print(main_branch)
                    #v = get_revisions_and_run_parser(repo, proj_name, main_branch)
                    if get_revisions_and_run_parser(repo, proj_name, main_branch) == -1:
                        print('no revision found')
                        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
                        continue
                    else:
                        get_revisions_and_run_parser(repo, proj_name, main_branch)

                    
                except Exception as e:
                    
                    f = open("//media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/plot_results/differences_nodes_edges_exceptions3.txt", "a")
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
main2("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")