import os
import sqlite3


def get_connection2():
    conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_test2.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_connection2_train():
    conn = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_train2.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_connection():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_train3.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_connection_test():
    conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_test3.db",isolation_level=None)
    cursor =  conn.cursor()
    return conn,cursor

def get_all_train_hashes():
    select_train_hashes = """SELECT hash from contents;"""
    val = []
    fin_resp = []
    conn,curr = get_connection()
    if conn != None:
         curr.execute(select_train_hashes)  
         val = curr.fetchall()
         fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
                     
    else:
        print("connection failed")
    conn.commit()
    #conn.close()
    return fin_resp

def get_all_test_hashes():
    select_test_hashes = """SELECT hash from contents_copy;"""
    val = []
    fin_resp = []
    conn,curr = get_connection_test()
    if conn != None:
         curr.execute(select_test_hashes)  
         val = curr.fetchall()
         fin_resp = [eac_val for each_cont in val if isinstance(val,list) and len(val) > 0 for eac_val in each_cont if isinstance(each_cont,tuple)]
                     
    else:
        print("connection failed")
    conn.commit()
    #conn.close()
    return fin_resp

def get_duplicate_hashes():
    all_test_hashes = get_all_test_hashes()
    print("length of test hashes", all_test_hashes)
    all_train_hashes = get_all_train_hashes()
    print("length of train hashes", all_train_hashes)
    duplicate_hashes = []

    for hash in all_test_hashes:
        if hash in all_train_hashes:
            duplicate_hashes.append(hash)
    return duplicate_hashes
    

def delete_duplicate_hashes():
    duplicate_hashes = get_duplicate_hashes()
    conn,curr = get_connection_test()
    if len(duplicate_hashes) > 0:
        for hash in duplicate_hashes:
            delete_duplicate_test_value_statement =  """delete from contents_copy where hash = (?);"""
            
            if conn != None:
                curr.execute(delete_duplicate_test_value_statement,(hash,)) 
        conn.commit()
        conn.close()
    
    print("length of test hashes after deletion", len(get_all_test_hashes()))


delete_duplicate_hashes()