import os
import sqlite3
import hashlib

def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True


def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db",isolation_level=None)
    #conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_train2.db")
    cursor =  conn.cursor()
    return conn,cursor

def calculate_sha256(data):
    # Convert data to bytes if it’s not already
    if isinstance(data, str):
        data = data.encode()

    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256(data).hexdigest()

    return sha256_hash

def insert_into_authors_table(file_path):
    commit_sha = None
    authors_name = None
    authors_email = None
    committers_name= None
    commiters_email = None
    all_commits = get_all_commit_sha()
    print("allcommits", all_commits)

    with open(file_path,"r") as cm:
        lines  = cm.readlines()
        
        for line in lines:
            # convert data to string and assign to content column
            content = line.split(",")
            
            
            if len(content) == 2:
                commit_sha = content[0] if is_sha1(content[0]) else "None"
                authors_data = content[1]

                authors_details = authors_data.split("_COMMA_")

                
                if len(authors_details) == 4:
                    authors_name = calculate_sha256(authors_details[0].strip()) if len(authors_details[0]) > 0 else "None"
                    authors_email = calculate_sha256(authors_details[1].strip()) if len(authors_details[1]) > 0 else "None"
                    committers_name = calculate_sha256(authors_details[2].strip()) if len(authors_details[2]) > 0 else "None"
                    commiters_email = calculate_sha256(authors_details[3].strip()) if len(authors_details[3]) > 0 else "None"

                elif len(authors_details) == 3:
                    authors_name = calculate_sha256(authors_details[0].strip()) if len(authors_details[0]) > 0 else "None"
                    authors_email = calculate_sha256(authors_details[1].strip()) if len(authors_details[1]) > 0 else "None"
                    committers_name = calculate_sha256(authors_details[2].strip()) if len(authors_details[2]) > 0 else "None"
                    commiters_email = "None"
            
                elif len(authors_details) == 2:
                    authors_name = calculate_sha256(authors_details[0].strip()) if len(authors_details[0]) > 0 else "None"
                    authors_email = calculate_sha256(authors_details[1].strip()) if len(authors_details[1]) > 0 else "None"
                    committers_name = "None"
                    commiters_email = "None"
            
                elif len(authors_details) == 1:
                
                    authors_name =  calculate_sha256(authors_details[0].strip()) if len(authors_details[0]) > 0 else "None"
                    authors_email = "None"
                    committers_name = "None"
                    commiters_email = "None"
            
                else:
                    authors_name = "None"
                    authors_email = "None"
                    committers_name = "None"
                    commiters_email = "None"

            
            elif len(content) == 1:
                commit_sha  = content[0] if is_sha1(content[0]) else "None"
                authors_name = "None"
                authors_email = "None"
                committers_name  = "None"
                commiters_email = "None"
            
            else:
                continue
            
            print("authors name",authors_name)
            print("committers_name",committers_name)
            print("authors email", authors_email)
            print("committers email", commiters_email)
            
            insert_authors_data = """INSERT INTO authors (commit_sha,author_name,author_email,committer_name,committer_email) VALUES(?,?,?,?,?);"""
            
            conn,cur = get_connection()
            val = None
            
            if conn != None:
                if commit_sha not in all_commits:
                    cur.execute(insert_authors_data,(commit_sha,authors_name,authors_email,committers_name,commiters_email))   
                else:
                    continue            
            else:
                if val != None:
                    print("executed")
                print("connection failed")
            conn.commit()


def get_all_commit_sha():
    select_projects = """SELECT commit_sha from authors;"""
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
    #conn.close()
    return fin_resp 

insert_into_authors_table("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/authors/commitsha_authors_unique.csv")
#insert_into_authors_table("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/commit_authors.csv")