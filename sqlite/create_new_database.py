
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
connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all_final.db")
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


curs =  connection.cursor()


revision = """CREATE TABLE "Revisions" (
  "Project_Name" TEXT,
  "File" TEXT,
  "Revision" TEXT,
  "Commit_SHA" TEXT,
  "Commit_Date" TEXT,
  "Hash" TEXT,
  "Nodes" INTEGER,
  "Edges" INTEGER);"""
curs.execute(revision)

contents = """CREATE TABLE "Contents" (
  "Hash" TEXT,
  "Content" TEXT
);"""
curs.execute(contents)


curs.execute('''CREATE UNIQUE INDEX "ix_Hashes_index" ON "Contents" ("Hash");''')


# step 5: close 
connection.commit()
connection.close()
