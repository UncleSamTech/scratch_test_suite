import sqlite3
import pandas as pd

# step 1: load data file
#df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2_test.csv')
#df3 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/authors_hashed2_test.csv')
#df4 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique2_test.csv')
#df2 = pd.read_csv('/pd_parsed/csvs/project_file_revision_commitsha_commitdate_hash_nodes_edges_final.csv')

# step 2: clean data
#df.columns = df.columns.str.strip()
#df3.columns = df3.columns.str.strip()
#df4.columns = df4.columns.str.strip()
#df2.columns = df2.columns.str.strip()

# step 3: create/connect to database
connection = sqlite3.connect("scratch_revisions.db")

# step 4: load data file to sqlite
#df.to_sql("Projects", connection, if_exists='replace', index=False)
#df3.to_sql("Authors", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
revision_obj = connection.cursor()

revision_obj.execute("DROP TABLE IF EXIST Revisions")
revision_table = """CREATE TABLE Revisions (Project_Name,
 File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges); """

revision_obj.execute(revision_table)

#df2.to_sql("Revisions", connection, if_exists='replace', index=False)
# step 5: close connection
connection.close()