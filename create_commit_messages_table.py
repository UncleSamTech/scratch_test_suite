import os
import pandas as pd
import sqlite3

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

def insert_into_commit_messages(file_path):
    commit_sha = None
    commit_message = None
    all_commits = get_all_commit_sha()
    with open(file_path,"r",encoding="utf",errors="ignore") as cm:
        lines  = cm.readlines()
        
        for line in lines:
            complete_content = line.split(",")
            
            if len(complete_content) == 2:
                commit_sha = complete_content[0] if is_sha1(complete_content[0]) else "None"
                commit_message = complete_content[1] 
            else:
                commit_sha = complete_content[0] if is_sha1(complete_content[0]) else "None"
                commit_message = "None"
         
            insert_commit_message = """INSERT INTO Commit_Message (Commit_Sha, Commit_Message) VALUES(?,?);"""
            
            conn,cur = get_connection()
            val = None
            
            if conn != None:
                if commit_sha not in all_commits:
                    cur.execute(insert_commit_message,(commit_sha,commit_message))   
                else:
                    continue            
            else:
                if val != None:
                    print("executed")
                print("connection failed")
            conn.commit()

        
def get_all_commit_sha():
    select_projects = """SELECT commit_sha from commit_message;"""
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


insert_into_commit_messages("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_upd.csv")
