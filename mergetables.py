import os
import sqlite3

conn_train = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_train3.db",isolation_level=None)
cursor1 = conn_train.cursor()

conn_test = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_test3.db",isolation_level=None)
cursor2 = conn_test.cursor()

connection_new = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_analysis.db")
cursor_new =  connection_new.cursor()

#create a new table in the database
cursor_new.execute("""CREATE TABLE IF NOT EXISTS Revisions (
  "Project_Name" TEXT,
  "File" TEXT,
  "Revision" TEXT,
  "Commit_SHA" TEXT,
  "Commit_Date" TEXT,
  "Hash" TEXT,
  "Nodes" INTEGER,
  "Edges" INTEGER);""")

#copy data from first table
cursor1.execute("Select * from revisions;")
rows1 = cursor1.fetchall()
cursor_new.executemany("INSERT into Revisions (Project_Name,File,Revision,Commit_SHA,Commit_Date,Hash,Nodes,Edges) VALUES (?,?,?,?,?,?,?,?)",rows1)

#copy data from second database
cursor2.execute("Select * from revisions;")
rows2 = cursor2.fetchall()
cursor_new.executemany("INSERT into Revisions (Project_Name,File,Revision,Commit_SHA,Commit_Date,Hash,Nodes,Edges) VALUES (?,?,?,?,?,?,?,?)",rows2)

#commit the changes
connection_new.commit()
conn_train.close()
conn_test.close()
connection_new.close()