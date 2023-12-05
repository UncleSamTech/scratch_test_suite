import pandas as pd


def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

df_project = pd.read_csv("/pd_parsed/csvs/branch_name_CORRECT.csv")
commit_messages_file = "/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/commit_message_unique.csv"

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
df.to_csv("/pd_parsed/author_commit_message/commit_message/commit_messages.csv", index=False)


df = pd.read_csv("/pd_parsed/author_commit_message/commit_message/commit_messages.csv")
unique_df = df.drop_duplicates()
unique_df.to_csv("/pd_parsed/author_commit_message/commit_message/commit_messages_unique_with_project.csv", index=False)

# drop the project_name column
df = pd.read_csv("/pd_parsed/author_commit_message/commit_message/commit_messages_unique_with_project.csv")
df = df.drop(columns=['Project_Name'])
unique_df = df.drop_duplicates()
unique_df.to_csv("/pd_parsed/author_commit_message/commit_message/commit_messages_unique.csv", index=False)