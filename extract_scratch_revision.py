import os
# fix the path to the pdparser accordingly

import sys
import json
from pydriller.git import Git
import re
#import tempfile
from datetime import datetime
import ast
import subprocess
#from pathlib import Path
from scratch_parser import scratch_parser
import sqlite3
from unzip_scratch import unzip_scratch
import hashlib

'''
# fix the database location
connection = pysqlite3.connect("../../database.db")
cursor = connection.cursor()
cursor.execute('BEGIN TRANSACTION')
'''

def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_database6.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def strip_pattern(input_string):
    pattern = r'.*?\.sb3\b'
    #pattern = r'\S*\.sb3\b'
    matches = re.findall(pattern,input_string)
    return matches

def store_repl_matches(matches):
    new_val = {}
    i = 0
    for each_match in matches:
        i += 1
        new_val[f'{each_match}#{i}'] = f'TEMP{i}'
    return new_val
        
def replace_parts(inp_str,inp_dict):
    if isinstance(inp_dict,dict):
        for k,v in inp_dict.items():
            k = k.split("#")[0]
            inp_str = inp_str.replace(k,v)
    return inp_str
'''
def replace_null_bytes_correct(byte_val):
    decoded_strings = []
    val  = re.split(b'[\xc2\x00]',byte_val)
    print('1',val)

    try:
        for item in val:
            decoded_string = item.decode('utf-8')
            decoded_strings.append(decoded_string)
    except UnicodeDecodeError:
        print(f"Error decoding byte value : {item}")

    print('2',decoded_strings)
    val3  = [v2+"#" for v2 in decoded_strings]
    print('3',val3)
    val4 = ''.join(val3)
    print('4',val4)
    val_enc = val4.encode('utf-8')
    print('5',val_enc)
    #decode = b'#'.join(val2)
 
    return val_enc
'''

def replace_list(inp_list,inp_dict):
    if isinstance(inp_dict,dict) and isinstance(inp_list,list):
        for k,v in inp_dict.items():
            k = k.split("#")[0]
            
                
            inp_list[0] = inp_list[0].replace(v,k)
    return inp_list

def replace_null_byte(byte_val,replace_byte):
    null_byte = b'x00'
    str_rep_byte = repr(byte_val.replace(null_byte,replace_byte))
    print(type(str_rep_byte))
    return str_rep_byte
    
def quick_convert(input_sequence):
    inp_strin = input_sequence.decode('utf-8') if isinstance(input_sequence,bytes) else input_sequence

    sed_comm = f"{inp_strin} | sed 's/\\x00/\\x2d/g'"
    print("sed com",sed_comm)
    res_str = subprocess.check_output(sed_comm,shell=True, stderr=subprocess.PIPE).decode('utf-8')
    print("res str",res_str)
    res_seq  = res_str.encode('utf-8')
    return res_seq

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
        
        print(f"original {byte_val} after decode {decoded_byte} after replacement {replace_val} final value {encoded_byte}")
      
        return encoded_byte

def count_seperator(fn):
    if isinstance(fn,str):
        #return fn.strip().count("#")
        return fn.strip().count("¬")
       
def correct_code_replace2(byte_val):
  
    inp = byte_val.replace(b'\xc2',b'#') if b'\xc2' in byte_val else byte_val
    return inp

def get_connection2():
    conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_database.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor


def escape_special_chars(input_string):
    escaped_string = subprocess.check_output(['echo', input_string], universal_newlines=True)
    escaped_string = subprocess.check_output(['sed', 's/[\\\/^$.*+?()[\]{}|[:space:]]/\\\\&/g'], input=escaped_string, universal_newlines=True)
    return escaped_string.strip()


def get_all_projects_in_db():
    select_projects = """SELECT Project_Name from revisions;"""
    val = []
    fin_resp = []
    conn,curr = get_connection2()
    if conn != None:
         curr.execute(select_projects)  
         val = curr.fetchall()
         fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
                     
    else:
        print("connection failed")
    conn.commit()
    return fin_resp

def calculate_sha256(data):
    # Convert data to bytes if it’s not already
    if isinstance(data, str):
        data = data.encode()

    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256(data).hexdigest()

    return sha256_hash 
def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def split_file(file_byte):
    if isinstance(file_byte,bytes):
        return file_byte.decode().strip().split('\n')

def get_revisions_and_run_parser(cwd, main_branch,project_name,  debug=False):
    sp = scratch_parser()
    #un = unzip_scratch()
    proc1 = subprocess.run(['git --no-pager log --pretty=tformat:"%H" {} --no-merges'.format(main_branch)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
    proc2 = subprocess.run(['xargs -I{} git ls-tree -r --name-only {}'], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc3 = subprocess.run(['grep -i "\.sb3$"'], input=proc2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    proc4 = subprocess.run(['sort -u'], input=proc3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
    
    filenames = proc4.stdout.decode().strip().split('\n')
    

    if filenames is None or filenames  == [''] or len(filenames) == 0 or filenames == []:
        #logging.error(f'no sb3 file found in {project_name} due to {logging.ERROR}')
        return -1
    else:
    # for all sb3 files in ths project
        for f in filenames:
            #f = escape_special_chars(f)
            proc1 = subprocess.run(['git --no-pager log -z --numstat --follow --pretty=tformat:"{}¬%H" -- "{}"'.format(f,f)], stdout=subprocess.PIPE, cwd=cwd, shell=True)
            
            proc2 = subprocess.run(["cut -f3"], input=proc1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            
            
            proc3_ = correct_code_replace(proc2.stdout)
            print("inside view",proc3_)
            #proc_main3 = correct_code_replace2(proc3_)
            proc4 = subprocess.run(['xargs -0 echo'], input=proc3_, stdout=subprocess.PIPE, cwd=cwd, shell=True)
            print("inside view decode", proc4.stdout)
            decoded_proc4 = proc4.stdout.decode()
            print("decoded_values parsed",decoded_proc4)
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
                
                print("filename",fn)
                separator_count = fn.strip().count("¬")
                #separator_count = fn.strip().count("#")
                #split_line = rep_str.strip("-").split('-')
                
                
                
            
                split_line = fn.strip('¬').split('¬')
                #split_line = fn.strip('#').split('#')
                
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
                #elif fn[0] == '#':
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
            

                commit_date = subprocess.run(['git log -1 --format=%ci {}'.format(c)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=cwd, shell=True).stdout.decode()
                parsed_date = datetime.strptime(commit_date.strip(), '%Y-%m-%d %H:%M:%S %z')
                parsed_date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S %z')


            # with open("/pd_parsed/csvs/project_file_revision_commitsha_commitdate_1.txt", "a") as outfile:
            #   ß  outfile.write("{}_COMMA_{}_COMMA_{}_COMMA_{}_COMMA_{}\n".format(project_name, f, new_name, c, parsed_date_str))


                file_contents = ''
            
                try:
                    contents1 = subprocess.run(['git show {}:"{}"'.format(c, new_name)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=cwd, shell=True)
                    '''
                    contents2 = subprocess.run(["sed 's/\\;/_SLASH_SEMICOLON_/g'"], input=contents1.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    contents3 = subprocess.run(["sed 's/;#X/;\\n#X/g'"], input=contents2.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    contents4 = subprocess.run(["sed 's/;#N/;\\n#N/g'"], input=contents3.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    contents5 = subprocess.run(["sed 's/; #X/;\\n#X/g'"], input=contents4.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    contents6 = subprocess.run(["sed 's/; #N/;\\n#N/g'"], input=contents5.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    contents7 = subprocess.run(["sed 's/_SLASH_SEMICOLON_/\\;/g'"], input=contents6.stdout, stdout=subprocess.PIPE, cwd=cwd, shell=True)
                    '''
                    #file_contents = contents1.stdout.decode("utf-8", "ignore")
                    val34 = contents1.stdout
                    if val34 is None:
                        continue
                    val = sp.decode_scratch_bytes(val34)
            
                    file_contents = val
                    
            
                    stats = sp.parse_scratch(file_contents,new_name)
                except:
                    stats = {"parsed_tree":[],"stats":{}}
            
            
                
                nodes_count = 0 
                edges_count = 0
                try:
                    nodes_count = stats["stats"]["number_of_nodes"]
                    edges_count = stats["stats"]["number_of_edges"]
                except:
                    nodes_count = 0
                    edges_count = 0
                '''
                try:
                    node = stats["nodes"]
                    edge = stats["edges"]
                except:
                    node = 0
                    edge = 0
                '''
                json_output = json.dumps(stats, indent=4)
                hash_value = calculate_sha256(str(json_output))
            
                new_original_file_name = f.replace(",", "_COMMA_")
                new_name = new_name.replace(",", "_COMMA_")
            
                insert_revision_statement = """INSERT INTO Revisions (Project_Name, File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges) VALUES(?,?,?,?,?,?,?,?);"""
                insert_hash_statement = """INSERT INTO Contents (Hash,Content) VALUES(?,?);"""
                tree_value = str(json_output)
                conn,cur = get_connection2()
                val = None
                if conn != None:
                    cur.execute(insert_revision_statement,(project_name,new_original_file_name,new_name,c,parsed_date_str,hash_value,nodes_count,edges_count))
                    cur.execute(insert_hash_statement,(hash_value,tree_value))
                    
                else:
                    if val != None:
                        print("executed")
                    print("connection failed")
                    conn.commit()
            '''
            cursor.execute("INSERT INTO Revisions (Project_Name, File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges) VALUES(?,?,?,?,?,?,?,?)", (project_name, new_original_file_name, new_name, c, parsed_date_str, hash_value, node, edge))
            cursor.execute("INSERT INTO Contents (Hash, Content) VALUES(?,?) ON CONFLICT(Hash) DO NOTHING", (hash_value, str(json_output)))
            '''
        return 1

        
'''
def main(filename: str):
    project_name, main_branch = filename.split(',')
    print(project_name)
    git_object = Git(f'pd_mirrored_extracted/{project_name}')
    git_object.checkout(main_branch)
    try:
        get_revisions_and_run_parser(f'pd_mirrored_extracted/{project_name}', project_name, main_branch)
    except Exception as e:
        print(e)
    connection.commit()
    connection.close()
'''
def main2(project_path: str):
    proj_names = []
    for i in os.listdir(project_path):
        if len(i) > 1 and os.path.isdir(f'{project_path}/{i}'):
            proj_names.append(i)
        else:
            continue
    projects_to_skip = get_all_projects_in_db()
    
    for proj_name in proj_names:
        
        if proj_name not in projects_to_skip and proj_name != '' and len(proj_name) > 1:
        #if  proj_name != '' and len(proj_name) > 1:
            repo = f'{project_path}/{proj_name}'
            main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
            main_branch = main_branch.stdout.decode("utf-8").strip('/n')[0:]
            
            
            if len(main_branch) > 1 or main_branch != '' or main_branch != None and repo != '' or repo != None and len(repo) > 0 and len(main_branch) > 0:
                try:
                    
                    
                    if get_revisions_and_run_parser(repo, main_branch,proj_name) == -1:
                        print('no revision found')
                        
                        continue
                    else:
                        print('found')
                        get_revisions_and_run_parser(repo, main_branch,proj_name)

                    


                except Exception as e:
                    
                    #f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/exceptions4.txt", "a")
                    f = open("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/repos/exceptions4.txt","a")
                    f.write("{}\n".format(e))
                    f.close()
                    
            else:
                print("skipped or visited")
                continue
        else:
            print(f"skipped {proj_name}")
            continue

'''
if __name__ == "__main__":
    filename = sys.argv[1]
    main(filename)
'''

#main2("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/repos")
#main2("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")
#quick_convert(b'CS50 - Problem Set 0 v2 (1).sb3\xc2\xac83143c732cdf6bc646d32701b9b1fb9c6ec3bf6a\x00\nCS50 - Problem Set 0 v2 (1).sb3\x00\n')