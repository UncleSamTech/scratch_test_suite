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
    train_project,test_project = train_test_split(data,test_size=ratio_test,train_size=ratio_train,random_state=42)
    print(f"total train set {len(train_project)} and total test set {len(test_project)}")
    return train_project,test_project


sample_train_test(get_all_project_names(),0.1)