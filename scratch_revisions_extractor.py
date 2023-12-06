import os
import sys
import json
from pydriller.git import Git
import tempfile
from datetime import datetime
import hashlib
import subprocess
from unzip_scratch import unzip_scratch
from pathlib import Path
from scratch_parser import scratch_parser
import logging
import sqlite3
import pysqlite3


def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions.db")
cursor = connection.cursor()
#print("total connection changes", connection.total_changes())
#cursor.execute('BEGIN TRANSACTION')

def calculate_sha256(content):
    # Convert data to bytes if it’s not already
    if isinstance(content, str):
        content = content.encode()

    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256(content).hexdigest()
    return sha256_hash


def get_revisions_and_run_parser(cwd, project_name, main_branch, debug=False):
    sp = scratch_parser()
    un = unzip_scratch()
    json_output = ''
    proc1 = subprocess.run(['git --no-pager log --pretty=tformat:"%H" {} --no-merges'.format(main_branch)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc2 = subprocess.run(['xargs -I{} git ls-tree -r --name-only {}'], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc3 = subprocess.run(['grep -i ".sb2$\|.sb3$"'], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True) 
    
    proc4 = subprocess.run(['sort -u'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    filenames = proc4.stdout.decode().strip().split('\n')
    print(filenames)
    if filenames is None or filenames  == [''] or len(filenames) == 0 or filenames == []:
        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
        return -1

    else:
        
        # for all sb3 files in ths project
        for f in filenames:
            proc1 = subprocess.run(['git --no-pager log -z --numstat --follow --pretty=tformat:"{}¬%H" -- "{}"'.format(f,f)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc2 = subprocess.run(["cut -f3"], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc3 = subprocess.run(["sed 's/\d0/¬/g'"], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc4 = subprocess.run(['xargs -0 echo'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            filename_shas = proc4.stdout.decode().strip().split('\n')
            filename_shas = [x for x in filename_shas if x != '']

        #if 2 ¬ then it is the original filename that we are trying to trace back (includes original filename, commit)
        #if 3 ¬ then it is not renamed (includes renamed filename, original filename, commit)
        #if starts with ¬ then it is renamed (includes pre-filename, renamed filename, original filename, commit)
        #if 1 ¬ then it is a beginning file with no diff; skip

            proc1 = subprocess.run(['git --no-pager log --all --pretty=tformat:"%H" -- "{}"'.format(f)], stdout=subprocess.PIPE, cwd=cwd, shell=True) # Does not produce renames
            all_shas = proc1.stdout.decode().strip().split('\n') 
            all_shas = [x for x in all_shas if x != '']
            all_sha_names = {}
        
            for x in all_shas:
                all_sha_names[x] = None

        # get filenames for each commit
            for fn in filename_shas: # start reversed, oldest to newest
                separator_count = fn.strip().count('¬')
                split_line = fn.strip('¬').split('¬')
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
            
                elif fn[0] == '¬':
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
            
            all_sha_dates = {}
            for c in all_sha_names.keys():
                commit_date = subprocess.run(['git log -1 --format=%ci {}'.format(c)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=cwd, shell=True).stdout.decode()
                parsed_date = datetime.strptime(commit_date.strip(), '%Y-%m-%d %H:%M:%S %z')

                all_sha_dates[c] = parsed_date

                # fill in the gaps
                prev_fn = f
            for c in all_sha_names.keys():
                if all_sha_names[c] is None:
                    all_sha_names[c] = prev_fn
                prev_fn = all_sha_names[c]
        
            stats = {}

            for c in sorted(all_sha_dates, key=all_sha_dates.get): # start oldest to newest
           
                new_name = all_sha_names[c]
            
                print(c)
                commit_date = subprocess.run(['git log -1 --format=%ci {}'.format(c)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=cwd, shell=True).stdout.decode()
                parsed_date = datetime.strptime(commit_date.strip(), '%Y-%m-%d %H:%M:%S %z')
                parsed_date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S %z')
                form_file = "{}_COMMA_{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, new_name, c, parsed_date_str)
                print(form_file)
                
                #with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/project_file_revision_commitsha_commitdate_alter.txt", "a") as outfile:
                    #outfile.write(form_file) 
                    


                file_contents = ''

                contents1 = subprocess.run(['git show {}:"{}"'.format(c, new_name)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)
                #print('contents1',contents1)
                val = sp.decode_scratch_bytes(contents1.stdout)
            
                file_contents = val
            
            
                stats = sp.parse_scratch(file_contents,new_name)
                print(stats)
            
                stats["commit_date"] = parsed_date_str
                stats["commit_sha"] = c
                
                json_output = json.dumps(stats, indent=4)
                hash_value = calculate_sha256(str(json_output))
                nodes_count = 0 
                edges_count = 0
                try:
                    filename_key = os.path.splitext(f)[0] if ".sb3" in f else f
                    filename_key = f'{filename_key}_summary'
                    print('node_egde_key',filename_key)
                    nodes_count = stats["stats"][filename_key]["number_of_nodes"]
                    edges_count = stats["stats"][filename_key]["number_of_edges"]
                except:
                    nodes_count = 0
                    edges_count = 0
                
                print('nodes_count',nodes_count)
                print('edges_count',edges_count)

                new_original_file_name = f.replace("/", "_FFF_")
                root_name = Path(new_original_file_name).stem
                
                #insert revisions and hashes to database
                #cursor.execute("""INSERT INTO Revisions VALUES({project_name},{new_original_file_name},{new_name},{c},{parsed_date_str},{hash_value},{nodes_count},{edges_count})""")
                #cursor.execute("INSERT INTO Hashes (Hash,Content) VALUES(?,?) ON CONFLICT(Hash) DO NOTHING",(hash_value),str(json_output))

                cursor.execute("INSERT INTO Revisions (Project_Name, File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges) VALUES(?,?,?,?,?,?,?,?))",(project_name,new_original_file_name,new_name,c,parsed_date_str,hash_value,nodes_count,edges_count))
                cursor.execute("INSERT INTO Hashes (Hash,Content) VALUES(?,?) ON CONFLICT(Hash) DO NOTHING",(hash_value),str(json_output))
                # suggestion: save the original file name extension here to avoid manual fixes later :(
            
                #com = f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/project2'
                #com = com + "/" +  project_name + "/" + new_original_file_name + "_CMMT_" + c + ".json"
                #print(com)
            
                #with open(com,"w") as outfile:
                    #outfile.write(json_output)
        return 1
            


def main(filename: str):
    lines = None
    count = 0
    with open(filename) as f:
        for lines in f:
            val = len(lines.split(','))
            if val == 2 and lines.split(',')[0] != '' and lines.split(',')[1] != '' and len(lines.split(',')[0]) > 0 and len(lines.split(',')[1]) > 0:

                project_name, main_branch = lines.split(',') 
                print(project_name, main_branch)
                print(len(project_name), len(main_branch))
                if project_name != '' and main_branch  != '' and len(project_name) > 1 and len(main_branch) > 1:
                #get_revisions_and_run_parser(f'/mnt/c/Users/USER/documents/scratch_tester/scratch_test_suite/files/repos/{project_name}', project_name, main_branch)
                    print("running")
                    print(project_name)
                    print(main_branch)
                    git_object = Git(f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted_test/{project_name}')
            
                    git_object.checkout(main_branch.strip())
        
                    try:
        
                        v = get_revisions_and_run_parser(f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/{project_name}', project_name, main_branch)
                        if v == -1:
                        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
                            continue
        
                    except Exception as e:
            
                        f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/exceptions.txt", "a")
                        f.write("{}\n".format(e))
                        f.close()
                    #logging.error(f'skipped {project_name}  to {logging.ERROR}')
                        pass
                    finally:
                        print("done")
                else:
                    print("skipped")
                    continue
            else:
                print("skipped")
                continue

def main2(project_path: str):
    proj_names = []
    for i in os.listdir(project_path):
        if len(i) > 1 and os.path.isdir(f'{project_path}/{i}'):
            proj_names.append(i)
        else:
            continue
    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/revisions_projectnames2.txt","w") as pna:
        if len(proj_names) > 0:
            for i in proj_names:
                pna.write("{}\n".format(i))
    for proj_name in proj_names:
        if proj_name != '' and len(proj_name) > 1:
            repo = f'{project_path}/{proj_name}'
            main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
            main_branch = main_branch.stdout.decode("utf-8").strip('/n')[0:]
            if len(main_branch) > 1 or main_branch != '' or main_branch != None and repo != '' or repo != None and len(repo) > 0 and len(main_branch) > 0:
                try:
                    #print(repo)
                    #print(proj_name)
                    #print(main_branch)
                    v = get_revisions_and_run_parser(repo, proj_name, main_branch)
                    if v == -1:
                        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
                        continue
                except Exception as e:
                    f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/exceptions2.txt", "a")
                    f.write("{}\n".format(e))
                    f.close()
                    #logging.error(f'skipped {project_name}  to {logging.ERROR}')
                    pass
                finally:
                    print("done")
                
                #connection.close()
            else:
                print("skipped")
                continue
        else:
            print("skipped")
            continue
    connection.commit()
    connection.close()
main2("/media/crouton/siwuchuk/newdir/vscode_repos_files/extracted_test")

