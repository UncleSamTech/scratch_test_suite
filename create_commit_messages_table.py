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
         
            insert_commit_message = """INSERT INTO Commit_Messages (Commit_Sha, Commit_Message) VALUES(?,?);"""
            
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

def insert_into_commit_messages_optimized(file_path):
    # Get all commit SHAs from the database at once
    all_commits = get_all_commit_sha()

    # Establish the database connection once
    conn, cur = get_connection()

    # Check if connection was established
    if conn is None:
        print("Connection failed")
        return

    with open(file_path, "r", encoding="utf-8", errors="ignore") as cm:
        # Prepare insert statement
        insert_commit_message = """INSERT INTO Commit_Messages (Commit_Sha, Commit_Message) VALUES (?, ?);"""
        try:
            # Read the file line by line
            for line in cm:
                # Split the line by comma
                complete_content = line.split(",")
                
                # Process the line based on its length
                commit_sha = complete_content[0].strip() if is_sha1(complete_content[0].strip()) else "None"
                commit_message = complete_content[1].strip() if len(complete_content) == 2 else "None"
                
                # Insert into DB only if commit_sha is not already in all_commits
                if commit_sha not in all_commits:
                    cur.execute(insert_commit_message, (commit_sha, commit_message))

            # Commit all changes at once
            conn.commit()
        except Exception as e:
            print(f"Error processing file: {e}")
        finally:
            # Close the connection
            conn.close()
        
def get_all_commit_sha():
    select_projects = """SELECT commit_sha from commit_messages;"""
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


insert_into_commit_messages_optimized("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_upd.csv")
