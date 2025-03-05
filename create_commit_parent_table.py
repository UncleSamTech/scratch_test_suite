import sqlite3
import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df_comm_par = pd.read_csv('/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/all_parent_commits/parent_commits_result_upd_filtered.csv')

# Step 2: Connect to the SQLite database
connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_cons_all.db")
c = connection.cursor()

# Step 3: Write the DataFrame to the SQLite table
df_comm_par.to_sql("Commit_Parentss", connection, if_exists='replace', index=False)

# Step 4: Commit the changes
connection.commit()

# Step 5: Create an index on the "Commit_SHA" column
c.execute('''CREATE INDEX "ix_Commit_Parents_index" ON "Commit_Parentss" ("Commit_SHA");''')

# Step 6: Commit the changes again
connection.commit()

# Step 7: Close the connection
connection.close()