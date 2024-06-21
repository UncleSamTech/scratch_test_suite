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
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def insert_into_commit_messages(file_path):
    commit_sha = None
    commit_message = None
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
                cur.execute(insert_commit_message,(commit_sha,commit_message))               
            else:
                if val != None:
                    print("executed")
                print("connection failed")
            conn.commit()

        
    


insert_into_commit_messages("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_unique_cleaned.csv")
