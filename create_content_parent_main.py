import sys
import json
from datetime import datetime
import subprocess
from pathlib import Path
import sqlite3
import os

#conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database.db')
#conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db')
conn = sqlite3.connect('/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db')

def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def count_seperator(fn):
    if isinstance(fn,str):
        #return fn.strip().count("#")
        return fn.strip().count("¬")

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

def is_valid_encoding(byte_string,encoding='utf-8'):
    try:
        decoded_string = byte_string.decode(encoding)
        print(type(decoded_string))
        return True
    except UnicodeDecodeError:
        print(type(decoded_string))
        return False

def get_parents_from_database(c):
    
    cursor = conn.cursor()
    cursor.execute("SELECT Parent_SHA FROM Commit_Parentss WHERE Commit_SHA = (?)", (c,))
    all_parents_of_c = cursor.fetchall()
    cursor.close()
    all_parents_of_c = set([x[0] for x in all_parents_of_c])
    return all_parents_of_c


def get_valid_parents_recursive(c, parents_of_c, commits_which_modified_file_f, visited_parents):

    parents = get_parents_from_database(c)
    common_parents = parents & commits_which_modified_file_f
    parents_of_c.update(common_parents)
    parents_not_in_common = parents - commits_which_modified_file_f

    for parent in parents_not_in_common:
        if (parent != "None") and (parent not in visited_parents):
            visited_parents.add(parent)
            get_valid_parents_recursive(parent, parents_of_c, commits_which_modified_file_f, visited_parents)


def get_connection_val():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

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
            #proc3 = subprocess.run(["sed 's/\d0/¬/g'"], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            proc3 = correct_code_replace(proc2.stdout)
            proc4 = subprocess.run(['xargs -0 echo'], input=proc3, stdout=subprocess.PIPE, cwd=cwd, shell=True)
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
                separator_count = count_seperator(fn)
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
 


            prev_fn = f
            for c in all_sha_names.keys():
                if all_sha_names[c] is None:
                    all_sha_names[c] = prev_fn
                prev_fn = all_sha_names[c]
        

            commits_which_modified_file_f = set(all_sha_names.keys()) # commits across all branches which modified file f
        
            for c in commits_which_modified_file_f:

                parents_of_c = set()
                visited_parents = set()
                # If there is only one commit that modified this file,then this is the one and there are no parents
                if len(commits_which_modified_file_f) > 1: 
                    get_valid_parents_recursive(c, parents_of_c, commits_which_modified_file_f, visited_parents)
            
                write_content_parents_opt(project_name,f,c,parents_of_c)
                # if len(parents_of_c) == 0:
                #     with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/content_parents_optimized_upd.csv", "a") as outfile:
                #         outfile.write("{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, c, c))
                # else:
                #     # we have a set of valid parents for c Get the node and edge count at each of these parents
                #     for parent in parents_of_c:

                #         with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/content_parents_optimized_upd.csv", "a") as outfile:
                #             outfile.write("{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, c, parent))

        return 1

def write_content_parents(project_name, f, c, parents_of_c):
    # Path to the output file
    output_file_path = "/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/content_parents_optimized_upd_2.csv"
    
    # Initialize a set to track unique lines, starting with the current file contents
    written_lines = set()

    # Read existing lines in the file and add them to the set
    try:
        with open(output_file_path, "r") as infile:
            for line in infile:
                written_lines.add(line.strip())  # Strip newline characters
    except FileNotFoundError:
        # If the file doesn't exist yet, we just continue with an empty set
        pass

    # Prepare lines to be written based on parents_of_c
    if len(parents_of_c) == 0:
        line = "{}_COMMA_{}_COMMA_{}_COMMA_{}".format(project_name, f, c, c)
        if line not in written_lines:
            with open(output_file_path, "a") as outfile:
                outfile.write(line + "\n")
            written_lines.add(line)
    else:
        # Loop through valid parents and write unique lines
        for parent in parents_of_c:
            line = "{}_COMMA_{}_COMMA_{}_COMMA_{}".format(project_name, f, c, parent)
            if line not in written_lines:
                with open(output_file_path, "a") as outfile:
                    outfile.write(line + "\n")
                written_lines.add(line)

def write_content_parents_opt(project_name, f, c, parents_of_c):
    # Path to the output file
    output_file_path = "/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/content_parents_optimized_upd_2.csv"
    
    # Initialize a set to track unique lines
    written_lines = set()

    # Read existing lines in the file and add them to the set
    try:
        with open(output_file_path, "r") as infile:
            written_lines.update(line.strip() for line in infile)
    except FileNotFoundError:
        # If the file doesn't exist yet, start with an empty set
        pass

    # Prepare lines to be written based on parents_of_c
    lines_to_write = []
    if not parents_of_c:
        line = f"{project_name}_COMMA_{f}_COMMA_{c}_COMMA_{c}"
        if line not in written_lines:
            lines_to_write.append(line)
            written_lines.add(line)
    else:
        for parent in parents_of_c:
            line = f"{project_name}_COMMA_{f}_COMMA_{c}_COMMA_{parent}"
            if line not in written_lines:
                lines_to_write.append(line)
                written_lines.add(line)

    # Write all new lines to the file at once
    if lines_to_write:
        with open(output_file_path, "a") as outfile:
            outfile.write("\n".join(lines_to_write) + "\n")


def main2(project_path: str):
    proj_names = []
    for i in os.listdir(project_path):
        if len(i) > 1 and os.path.isdir(f'{project_path}/{i}'):
            print(project_path)
            proj_names.append(i)
        else:
            continue
   
    for proj_name in proj_names:
        if proj_name != '' and len(proj_name) > 1:
            repo = f'{project_path}/{proj_name}'
            main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
            main_branch = main_branch.stdout.decode("utf-8").strip('/n')[0:]
            
            if len(main_branch) > 1 or main_branch != '' or main_branch != None and repo != '' or repo != None and len(repo) > 0 and len(main_branch) > 0:
                try:
                    get_revisions_and_run_parser(repo, proj_name, main_branch)
                    
                except Exception as e:
                    
                    f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/exceptions4.txt", "a")
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



def main2_optimized(project_path: str):
    proj_names = [i for i in os.listdir(project_path) if os.path.isdir(f'{project_path}/{i}') and len(i) > 1]
    
    for proj_name in proj_names:
        repo = f'{project_path}/{proj_name}'
        try:
            # Get the current branch name
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=False)
            main_branch = result.stdout.decode("utf-8").strip()

            # Check if main_branch and repo are valid
            if main_branch and repo:
                # Run the parser if branch name is valid
                get_revisions_and_run_parser(repo, proj_name, main_branch)
            else:
                print(f"Skipped project: {proj_name}")
        
        except Exception as e:
            # Log exceptions with proper error handling
            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/exceptions4.txt", "a") as f:
                f.write(f"Error with project {proj_name}: {e}\n")

def insert_into_content_parent_table(file_path):
    lines = None
    project_name= None
    file_name = None
    commit_sha = None
    content_sha = None

    with open(file_path,"r",encoding="utf-8") as cpd:
        lines = cpd.readlines()

        for each_line in lines:
            content = each_line.split("_COMMA_")
            
            if len(content) == 4:
                project_name = content[0].strip()
                file_name = content[1].strip()
                commit_sha = content[2].strip() if is_sha1(content[2]) else "None"
                content_sha = content[3].strip() if is_sha1(content[3].strip()) else "None"

            
            elif len(content) == 3:
                project_name = content[0].strip()
                file_name = content[1].strip()
                commit_sha = content[2].strip() if is_sha1(content[2]) else "None"
                content_sha = "None"
            
            elif len(content) == 2:
                project_name = content[0].strip()
                file_name = content[1].strip()
                commit_sha = "None"
                content_sha = "None"

            elif len(content) == 1:
                project_name = content[0].strip()
                file_name = "None"
                commit_sha = "None"
                content_sha = "None"
            
            else:
                project_name = "None"
                file_name = "None"
                commit_sha = "None"
                content_sha = "None"
            
            insert_into_content_parent = """INSERT INTO Content_Parents (Project_Name,File,Commit_SHA,Content_Parent_SHA) VALUES(?,?,?,?);"""

            conn,cur = get_connection_val()
            val = None
            
            if conn != None:
                cur.execute(insert_into_content_parent,(project_name,file_name,commit_sha,content_sha))               
            else:
                if val != None:
                    print("executed")
                print("connection failed")
            conn.commit()

def insert_into_content_parent_table_optimized(file_path):
    def parse_line(content):
        # Strip each part of the content
        content = [item.strip() for item in content]
        
        # Determine values based on the number of parts in the line
        project_name = content[0] if len(content) > 0 else "None"
        file_name = content[1] if len(content) > 1 else "None"
        commit_sha = content[2] if len(content) > 2 and is_sha1(content[2]) else "None"
        content_sha = content[3] if len(content) > 3 and is_sha1(content[3].strip()) else "None"
        
        return (project_name, file_name, commit_sha, content_sha)

    # Open connection once
    conn, cur = get_connection_val()
    if conn is None:
        print("Connection failed")
        return
    
    insert_into_content_parent = """
        INSERT INTO Content_Parentss (Project_Name, File, Commit_SHA, Content_Parent_SHA) 
        VALUES (?, ?, ?, ?);
    """
    
    # List to store parsed rows for batch insertion
    data_to_insert = []
    
    with open(file_path, "r", encoding="utf-8") as cpd:
        lines = cpd.readlines()

        for each_line in lines:
            content = each_line.split("_COMMA_")
            parsed_row = parse_line(content)
            data_to_insert.append(parsed_row)

    # Use executemany to batch insert all rows at once
    try:
        cur.executemany(insert_into_content_parent, data_to_insert)
        conn.commit()
        print(f"{len(data_to_insert)} rows inserted.")
    except Exception as e:
        print(f"Error occurred during insertion: {e}")
    finally:
        conn.close()

#main2_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")
insert_into_content_parent_table_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/content_parents/content_parents_optimized_upd_2.csv")



    