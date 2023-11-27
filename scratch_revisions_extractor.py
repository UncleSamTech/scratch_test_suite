import os
import sys
import json
from pydriller.git import Git
import tempfile
from datetime import datetime
import subprocess
from unzip_scratch import unzip_scratch
from pathlib import Path
from scratch_parser import scratch_parser
import logging

def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

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

                with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2_extracted_revisions/project_file_revision_commitsha_commitdate_1.txt", "a") as outfile:
                    outfile.write("{}_COMMA_{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, new_name, c, parsed_date_str))


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
            
                new_original_file_name = f.replace("/", "_FFF_")
                root_name = Path(new_original_file_name).stem
                # suggestion: save the original file name extension here to avoid manual fixes later :(
            
                com = f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2projects_mirrored_extracted/{project_name}/{root_name}_CMMT_{c}.json'
            
            
                with open(com,"w") as outfile:
                    outfile.write(json_output)
        return 1
            


def main(filename: str):
    lines = None
    count = 0
    with open(filename) as f:
        for lines in f:
            project_name, main_branch = lines.split(',')
            if project_name == '' or main_branch == '':
                count += 1
                logging.error(f"missing {project_name} or {main_branch} to {logging.ERROR} ")
                continue
            #get_revisions_and_run_parser(f'/mnt/c/Users/USER/documents/scratch_tester/scratch_test_suite/files/repos/{project_name}', project_name, main_branch)
            git_object = Git(f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2projects_mirrored_extracted/{project_name}')
    
            git_object.checkout(main_branch.strip())
        
            try:
        
                v = get_revisions_and_run_parser(f'/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2projects_mirrored_extracted/{project_name}', project_name, main_branch)
                if v == -1:
                    #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
                    continue
        
            except Exception as e:
            
                f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2_extracted_revisions/exceptions.txt", "a")
                f.write("{}\n".format(e))
                f.close()
                #logging.error(f'skipped {project_name}  to {logging.ERROR}')
                pass
            finally:
                print("done")
                print("missed", count)


main("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb2_branch_name_MIRROR.txt")

