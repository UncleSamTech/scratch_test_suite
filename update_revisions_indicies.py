import sqlite3
import pandas as pd

# step 1: create/connect to database
conn = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_analysis.db")
c = conn.cursor()

# Convert the "date_column" to datetime
c.execute("ALTER TABLE Revisions ADD COLUMN Commit_DateTime DATETIME;")

# Commit the changes
conn.commit()

c.execute("UPDATE Revisions SET Commit_DateTime = DATETIME(Commit_Date);")

conn.commit()

# create index on the hash column
c.execute('''CREATE INDEX "ix_Revisions_Hashes_index" ON "Revisions" ("Hash");''')
conn.commit()

c.execute('''CREATE INDEX "ix_Revisions_Projects_index" ON "Revisions" ("Project_Name");''')
conn.commit()

c.execute('''CREATE INDEX "ix_Revisions_Commit_index" ON "Revisions" ("Commit_SHA");''')
conn.commit()

conn.commit()
# Close the database connection
conn.close()