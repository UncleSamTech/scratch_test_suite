import os
import subprocess

def author_commit(path):
    
    if os.path.isdir(path):
        for i in os.listdir(path):
            if not os.path.isdir(f'{path}/{i}'):
                continue
            else:
                if not os.listdir(f'{path}/{i}'):
                    continue
                else:
                    try:
                        #with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/projnames2.txt","w") as wf:
                            #wf.write("{}\n".format(i))
                        subprocess.call(['sh', '/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/total_commit.sh'])
                    except Exception as e:
                        f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/total_commit_exceptions.txt", "a")
                        f.write("{}\n".format(e))
                        f.close()
        
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
                        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commits_result.csv","a") as pc:
                            pc.write(commit_sha + "," + "None\n")
                    else:
                        for p in parent_sha:
                            with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commits_result.csv","a") as pc:
                                pc.write(commit_sha + "," + f'{p}\n')
                                

                    
author_commit("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")
#parent_commit("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commit")