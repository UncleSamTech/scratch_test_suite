import os
import sqlite3
import pandas as pd

def create_projects(projects_path):
    
    for i in os.listdir(projects_path):
        if len(i) > 1 and os.path.isdir(f'{projects_path}/{i}'):
            #with open("all_projects_names.txt","a") as pn:
            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names.txt","a") as pn:
                pn.write(f"{i}\n")
        else:
            continue


def insert_into_projects_table(project_commit_branch_path):
    #LOAD the data
    df = pd.read_csv(project_commit_branch_path)

    #clean the data
    df.columns = df.columns.str.strip()

    #connect to database
    connection = sqlite3.connect("/Users/samueliwuchukwu/documents/scratch_database/scratch_revisions_main_train2.db")
    #connection = sqlite3.connect("/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/sqlite/scratch_revisions_main_all.db")
    #load data into file
    df.to_sql("Projects", connection, if_exists='replace', index=False)

#create_projects("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/repos")
#create_projects("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")
#/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects
#insert_into_projects_table("/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch.csv")
insert_into_projects_table("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects//media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names_commit_branch_cleaned.csv")