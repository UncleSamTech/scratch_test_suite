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
                        subprocess.call(['sh', '/media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/extract_parent_commit.sh'])
                    except Exception as e:
                        f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commit_exceptions.txt", "a")
                        f.write("{}\n".format(e))
                        f.close()
        
                    
author_commit("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")