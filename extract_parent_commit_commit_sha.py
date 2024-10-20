import os
def parent_commit(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            with open(os.path.join(path,i),'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    data = line.split(" ")
                    commit_sha = data[0]
                    parent_sha = data[1:]

                    if len(parent_sha) == 0:
                        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/all_parent_commits/parent_commits_result_upd.csv","a") as pc:
                            pc.write(commit_sha + "," + "None\n")
                    else:
                        for p in parent_sha:
                            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/all_parent_commits/parent_commits_result_upd.csv","a") as pc:
                                pc.write(commit_sha + "," + f'{p}\n')
                                
parent_commit("/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/parent_commits_upd")