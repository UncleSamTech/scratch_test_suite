import sqlite3
import pandas as pd

# step 1: load data file
#df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2.csv')
#df3 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/authors_hashed2.csv')
#df4 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique2.csv')
df5 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commits_result_unique.csv',sep="__SEP__")


# step 2: clean data
#df.columns = df.columns.str.strip()
#df3.columns = df3.columns.str.strip()
#df4.columns = df4.columns.str.strip()
df5.columns = df5.columns.str.strip()


# step 3: create/connect to database
connection = sqlite3.connect("scratch_revisions_database.db")

# step 4: load data file to sqlite
#df.to_sql("Projects", connection, if_exists='replace', index=False)
#df3.to_sql("Authors", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
df5.to_sql("Commit_Parents",connection,if_exists='replace',index=False)

#create the revision table to be used later
#revision_obj = connection.cursor()
#revision_table = """CREATE TABLE Revisions (Project_Name,File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges); """
#revision_obj.execute(revision_table)

#create the hash table to be used later
#hash_obj = connection.cursor()
#hash_table = """CREATE TABLE Hashes (Hash,Content); """
#hash_obj.execute(hash_table)


# step 5: close 
connection.commit()
connection.close()