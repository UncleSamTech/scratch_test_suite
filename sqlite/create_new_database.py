
import sqlite3
#import pandas as pd

# step 1: load data file
#df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/projects.csv')
#df3 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/authors_hashed2.csv')
#df4 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique2.csv')
#df5 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/content_parents/content_parents_unique1.csv')
#df_proj_load = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/authors/projectsnames_commitsha_authors_unique.csv')

# step 2: clean data
#df.columns = df.columns.str.strip()
#df3.columns = df3.columns.str.strip()
#df4.columns = df4.columns.str.strip()
#df5.columns = df5.columns.str.strip()


# step 3: create/connect to database
#connection = sqlite3.connect("scratch_revisions_database.db")
#connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all_final.db")
#connection = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_train2.db")
#connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_train3.db")
#connection_test = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_test3.db")
# step 4: load data file to sqlite
#df.to_sql("Projects", connection, if_exists='replace', index=False)
#df3.to_sql("Authors", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
#df5.to_sql("Content_Parents",connection,if_exists='replace',index=False)
#df_proj_load("Authors_Project",connection,if_exists='replace',index=False)

#c =  connection.cursor()
#cont_parent_table = """CREATE TABLE Content_Parents(Project_Name,File,Commit_SHA,Content_Parent_SHA);"""
#c.execute(cont_parent_table)

'''
revision = """CREATE TABLE "Revisions" (
  "Project_Name" TEXT,
  "File" TEXT,
  "Revision" TEXT,
  "Commit_SHA" TEXT,
  "Commit_Date" TEXT,
  "Hash" TEXT,
  "Nodes" INTEGER,
  "Edges" INTEGER
);"""
c.execute(revision)

contents = """CREATE TABLE "Contents" (
  "Hash" TEXT,
  "Content" TEXT
);"""
c.execute(contents)
connection.commit()

revision_hash_index="""CREATE INDEX "sc_Revisions_Hashes_index" ON "Revisions" ("Hash"); """
revision_project_index="""CREATE INDEX "sc_Revisions_Projects_index" ON "Revisions" ("Project_Name"); """
revision_commit_index="""CREATE INDEX "sc_Revisions_Commit_index" ON "Revisions" ("Commit_SHA"); """
project_project_name_index="""CREATE INDEX "sc_Projects_index" ON "Projects" ("Project_Name"); """
authors_author_index="""CREATE INDEX "sc_Authors_index" ON "Authors" ("Commit_SHA"); """
commit_messages_commit_sha_index="""CREATE INDEX "sc_Commit_Messages_index" ON "Commit_Messages" ("Commit_SHA"); """
commit_parents_commitsha_index="""CREATE INDEX "ix_Commit_Parents_index" ON "Commit_Parents" ("Commit_SHA"); """
content_parents_commit_sha_index="""CREATE INDEX "ix_Content_Parents_index" ON "Content_Parents" ("Commit_SHA"); """
'''

# step 5: close 

def merge_table_revisions(train_path,test_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch train and test db
  curs_all.execute(f"ATTACH '{train_path}' AS train_db")
  curs_all.execute(f"ATTACH '{test_path}' AS test_db")

  create_table = """CREATE TABLE IF NOT EXISTS Revisions (
  "Project_Name" TEXT,
  "File" TEXT,
  "Revision" TEXT,
  "Commit_SHA" TEXT,
  "Commit_Date" TEXT,
  "Hash" TEXT,
  "Nodes" INTEGER,
  "Edges" INTEGER
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Revisions (Project_Name, File, Revision,Commit_SHA,Commit_Date,Hash,Nodes,Edges)
    SELECT Project_Name, File, Revision,Commit_SHA,Commit_Date,Hash,Nodes,Edges from train_db.Revisions
    UNION ALL
    SELECT Project_Name, File, Revision,Commit_SHA,Commit_Date,Hash,Nodes,Edges from test_db.Revisions"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE train_db;")
  curs_all.execute("DETACH DATABASE test_db;")
  cons_db_conn.close()

def merge_table_hash(train_path,test_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch train and test db
  curs_all.execute(f"ATTACH '{train_path}' AS train_db")
  curs_all.execute(f"ATTACH '{test_path}' AS test_db")

  create_table = """CREATE TABLE IF NOT EXISTS Contents (
  "Hash" TEXT,
  "Content" TEXT
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Contents (Hash,Content)
    SELECT Hash,Content from train_db.Contents
    UNION ALL
    SELECT Hash,Content from test_db.Contents"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE train_db;")
  curs_all.execute("DETACH DATABASE test_db;")
  cons_db_conn.close()

def move_table_authors(authors_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch authors
  curs_all.execute(f"ATTACH '{authors_path}' AS authors_db")
  

  create_table = """CREATE TABLE IF NOT EXISTS Authors (
  "Commit_SHA" TEXT,
  "Author_Name" TEXT,
  "Author_Email" TEXT,
  "Committer_Name" TEXT,
  "Committer_Email" TEXT
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Authors (Commit_SHA,Author_Name,Author_Email,Committer_Name,Committer_Email) SELECT Commit_SHA,Author_Name,Author_Email,Committer_Name,Committer_Email from authors_db.Authors;"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE authors_db;")
  cons_db_conn.close()


def move_table_commit_message(commit_message_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch authors
  curs_all.execute(f"ATTACH '{commit_message_path}' AS commit_message_db")
  

  create_table = """CREATE TABLE IF NOT EXISTS Commit_Messages (
  "Commit_SHA" TEXT,
  "Commit_Message" TEXT
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Commit_Messages (Commit_SHA,Commit_Message) SELECT Commit_SHA,Commit_Message from commit_message_db.Commit_Message;"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE commit_message_db;")
  cons_db_conn.close()

def move_table_projects(projects_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch authors
  curs_all.execute(f"ATTACH '{projects_path}' AS projects_db")
  

  create_table = """CREATE TABLE IF NOT EXISTS Projects (
  "Project_Name" TEXT,
  "Default_Branch" TEXT,
  "Total_Commits" INTEGER
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Projects (Project_Name,Default_Branch,Total_Commits) SELECT Project_Name,Default_Branch,Total_Commits from projects_db.Projects;"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE projects_db;")
  cons_db_conn.close()

def move_table_commit_parents(commit_parents_path,cons_path):
  cons_db_conn = sqlite3.connect(cons_path)
  curs_all = cons_db_conn.cursor()

  #attatch authors
  curs_all.execute(f"ATTACH '{commit_parents_path}' AS commit_parents_db")
  

  create_table = """CREATE TABLE IF NOT EXISTS Commit_Parents (
  "Commit_SHA" TEXT,
  "Parent_SHA" TEXT
  );"""
  curs_all.execute(create_table)

  insert_statement = """INSERT INTO Commit_Parents (Commit_SHA,Parent_SHA) SELECT Commit_SHA,Commit_Parent from commit_parents_db.Commit_Parents;"""
  curs_all.execute(insert_statement)
  cons_db_conn.commit()
  curs_all.execute("DETACH DATABASE commit_parents_db;")
  cons_db_conn.close()



former_path = '/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db'
cons_path = '/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db'
move_table_commit_parents(former_path,cons_path)
#c.execute('''CREATE UNIQUE INDEX "ix_Hashes_index" ON "Contents" ("Hash");''')
#connection.commit()
#connection.close()
