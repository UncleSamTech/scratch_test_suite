import pandas as pd


def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

df_project = pd.read_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2.txt")
commit_messages_file = "/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/commit_message_2.csv"

project_names = df_project['Project_Name'].values

projects = []
commits = []
commit_messages = []


with open(commit_messages_file, 'r', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()

    for line in lines:
        # convert data to string and assign to content column
        content = line.split(",")
        project = content[0]
        try:
            commit = content[1]
        except:
            commit = ""
        if project not in project_names and is_sha1(commit) == False:
            commit_messages[-1] += line
        else:
            commit_message = content[2]
            projects.append(project)
            commits.append(commit)
            commit_messages.append(commit_message)

df = pd.DataFrame(list(zip(projects,commits, commit_messages)), columns =['Project_Name','Commit_SHA', 'Commit_Message'])
df['Commit_Message'] = df['Commit_Message'].str.rstrip('\n')
df.to_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages2.csv", index=False)


df = pd.read_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages2.csv")
unique_df = df.drop_duplicates()
unique_df.to_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique_with_project2.csv", index=False)

# drop the project_name column
df = pd.read_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique_with_project2.csv")
df = df.drop(columns=['Project_Name'])
unique_df = df.drop_duplicates()
unique_df.to_csv("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auth_commit_summary/summary/commit_messages_unique2.csv", index=False)