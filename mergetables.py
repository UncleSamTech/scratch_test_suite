import os
import sqlite3

conn_train = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db",isolation_level=None)
cursor1 = conn_train.cursor()

conn_test = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_test3.db",isolation_level=None)
cursor2 = conn_test.cursor()

connection_new = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_analysis.db")
cursor_new =  connection_new.cursor()

#create a new table in the database
cursor_new.execute("""CREATE TABLE IF NOT EXISTS Projects (
  "Project_Name" TEXT,
   "Total_Commits" INTEGER,
    "Default_Branch" TEXT
   );""")

#copy data from first table
cursor1.execute("Select * from Projects;")
rows1 = cursor1.fetchall()
cursor_new.executemany("INSERT into Projects (Project_Name,Total_Commits,Default_Branch) VALUES (?,?,?)",rows1)

#copy data from second database
#cursor2.execute("Select * from Contents;")
#rows2 = cursor2.fetchall()
#cursor_new.executemany("INSERT into Contents (Hash,Content) VALUES (?,?)",rows2)

print("done")
#commit the changes
connection_new.commit()
conn_train.close()
conn_test.close()
connection_new.close()