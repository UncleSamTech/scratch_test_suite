import os

#extract project names
def extract_all_projects(project_path,extracted_path):
    proj_names = []
    for i in os.listdir(project_path):
        if i and os.path.isdir(f'{project_path}/{i}'):
            proj_names.append(i)
            with open(extracted_path,"w") as proj_names:
                proj_names.write(f"{i}/n")
        else:
            continue
    

extract_all_projects("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted","/media/crouton/siwuchuk/newdir/vscode_repos_files/all_sb3_projects.csv")
