import os
import pandas as pd
import sqlite3

def insert_into_commit_messages(file_path):
    df = pd.read_csv(file_path)

    df = df.columns.str.strip()

    connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db")

    df.to_sql("Commit_Messages", connection, if_exists='replace', index=False)


insert_into_commit_messages("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_unique.csv")
