import os
import subprocess

def author_commit(path):
    all_projects = []
    if os.path.isdir(path):
        for i in os.listdir(path):
            if not os.path.isdir(f'{path}/{i}'):
                continue
            else:
                if not os.listdir(f'{path}/{i}'):
                    continue
                else:
                    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/projnames.txt","w") as wf:
                        wf.write("{}\n".format(i))
                    subprocess.call(['sh', '/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/author_commit_message.sh'])
        
                    
author_commit("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/projects")