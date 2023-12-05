import os
import subprocess

def create_non_empty_proj_branch(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            if not os.path.isdir(f'{path}/{i}'):
                continue
            else:
                if not os.listdir(f'{path}/{i}'):
                    continue
                else:
                    try:
                        repo = os.path.join(path,i)
                        main_branch = subprocess.run(['git rev-parse --abbrev-ref HEAD'], stdout=subprocess.PIPE, cwd=repo, shell=True)
                        main_branch = main_branch.stdout.decode("utf-8").strip('/n')[0:]
                    
                        with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/projectnames_branch_names.txt","a") as wp:
                            wp.write(i + "," + main_branch)
                    except Exception as e:
                        f = open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/project_exceptions.txt", "a")
                        f.write("{}\n".format(e))
                        f.close()

create_non_empty_proj_branch("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/projects")