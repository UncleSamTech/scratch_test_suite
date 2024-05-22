import sqlite3
import pandas as pd

# step 1: load data file
#df = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/projects.csv')
#df3 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/authors_hashed2.csv')
#df4 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique2.csv')
#df5 = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/content_parents/content_parents_unique1.csv')


# step 2: clean data
#df.columns = df.columns.str.strip()
#df3.columns = df3.columns.str.strip()
#df4.columns = df4.columns.str.strip()
#df5.columns = df5.columns.str.strip()


# step 3: create/connect to database
#connection = sqlite3.connect("scratch_revisions_database.db")
connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_train3.db")
connection_test = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_test3.db")
# step 4: load data file to sqlite
#df.to_sql("Projects", connection, if_exists='replace', index=False)
#df3.to_sql("Authors", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
#df4.to_sql("Commit_Messages", connection, if_exists='replace', index=False)
#df5.to_sql("Content_Parents",connection,if_exists='replace',index=False)

#create the revision table to be used later
revision_obj = connection.cursor()
revision_table = """CREATE TABLE Revisions (Project_Name,File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges); """
revision_obj.execute(revision_table)

#create the revision table on the test db to be used later
revision_obj_test = connection_test.cursor()
revision_table_test = """CREATE TABLE Revisions (Project_Name,File, Revision, Commit_SHA, Commit_Date, Hash, Nodes, Edges); """
revision_obj_test.execute(revision_table_test)


#create the hash table to be used later
hash_obj = connection.cursor()
hash_table = """CREATE TABLE Contents (Hash,Content); """
hash_obj.execute(hash_table)

#create the hash table on the test db to be used later
hash_obj_test = connection_test.cursor()
hash_table_test = """CREATE TABLE Contents (Hash,Content); """
hash_obj_test.execute(hash_table_test)

revision_hash_index="""CREATE INDEX "sc_Revisions_Hashes_index" ON "Revisions" ("Hash"); """
revision_project_index="""CREATE INDEX "sc_Revisions_Projects_index" ON "Revisions" ("Project_Name"); """
revision_commit_index="""CREATE INDEX "sc_Revisions_Commit_index" ON "Revisions" ("Commit_SHA"); """
project_project_name_index="""CREATE INDEX "sc_Projects_index" ON "Projects" ("Project_Name"); """
authors_author_index="""CREATE INDEX "sc_Authors_index" ON "Authors" ("Commit_SHA"); """
commit_messages_commit_sha_index="""CREATE INDEX "sc_Commit_Messages_index" ON "Commit_Messages" ("Commit_SHA"); """
commit_parents_commitsha_index="""CREATE INDEX "ix_Commit_Parents_index" ON "Commit_Parents" ("Commit_SHA"); """
content_parents_commit_sha_index="""CREATE INDEX "ix_Content_Parents_index" ON "Content_Parents" ("Commit_SHA"); """



#create the test_project
test_proj_obj = connection_test.cursor()
test_proj = """ CREATE TABLE test_projects(project_name,number); """
test_proj_obj.execute(test_proj)


#create the train_project
train_proj_obj = connection.cursor()
train_proj = """ CREATE TABLE train_projects(project_name,number); """
train_proj_obj.execute(train_proj)



# step 5: close 
connection.commit()
connection.close()

connection_test.commit()
connection_test.close()