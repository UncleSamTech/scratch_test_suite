import os
from sklearn.model_selection import train_test_split
import sqlite3

def get_all_hashes_from_projects(db_path):
    pass


def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_all_project_names():
    #make connection
    fin_resp = []
    conn,curs = get_connection()
    RETR_PROJ_QUERY = """SELECT project_name from projects;"""
    if conn != None:
         curs.execute(RETR_PROJ_QUERY)  
         val = curs.fetchall()
         fin_resp = [each_cont[0] for each_cont in val]                   
    else:
        print("connection failed")
    conn.commit()
    
    return fin_resp

def sample_train_test(data, ratio_train, ratio_test):
    train_project,_ = train_test_split(data, train_size=ratio_train, random_state=None)
    test_project, _ = train_test_split(data, train_size=ratio_test, random_state=None)

    print(f"total train set {len(train_project)} and total test set {len(test_project)}")
    return train_project,test_project

def retr_hash_match_project(project_name):
    hash_list = []
    conn,curs = get_connection()
    GET_HASHES = """SELECT hash FROM revisions WHERE project_name = ?;"""
    if conn != None:
        curs.execute(GET_HASHES,(project_name,))
        hashes = curs.fetchall()
        hash_list = [each_hash[0] for each_hash in hashes]
    else:
        print("connection failed")
    return hash_list

def retr_all_hash_for_proj_set(all_projects):
    all_hash = []
    if all_projects:
        for each_project in all_projects:
            each_project =  each_project.strip()
            res_hash = retr_hash_match_project(each_project)
            all_hash.extend(res_hash)
    return all_hash

def eliminate_duplicates_test_hashes(train_hashes, test_hashes):
    # Convert train_hashes to a set for fast lookup
    train_set = set(train_hashes)
    
    # Filter out hashes from test_hashes that are already in train_hashes
    unique_test_hashes = [hash_val for hash_val in test_hashes if hash_val not in train_set]
    
    return unique_test_hashes
    

train_proj,test_proj = sample_train_test(get_all_project_names(),0.1,0.2)
train = retr_all_hash_for_proj_set(train_proj)
print(f"total train {len(train)}")
test = retr_all_hash_for_proj_set(test_proj)
print(f"total test {len(test)}")
uniq_test = eliminate_duplicates_test_hashes(train,test)
print(f"total unique test {len(uniq_test)}")
