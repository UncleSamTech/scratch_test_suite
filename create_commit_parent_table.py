import sqlite3
import pandas as pd

df_comm_par = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/all_parent_commits/parent_commits_result_unique.csv')

connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db")

c = connection.cursor()

df_comm_par.to_sql("Commit_Parents",connection,if_exists='replace',index=False)
c.execute('''CREATE INDEX "ix_Commit_Parents_index" ON "Commit_Parents" ("Commit_SHA");''')

# step 5: close connection
connection.commit()
connection.close()