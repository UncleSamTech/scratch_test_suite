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
    print(fin_resp)
    return fin_resp


get_all_project_names()