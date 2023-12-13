import os

def get_folders(path):
    proj_names = []
    for i in os.listdir(path):
        full_path = f'{path}/{i}'
        if len(i) > 1 and len(full_path) > 1 and os.path.isdir(full_path):
            proj_names.append(i)
        else:
            continue
    with open("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/revisions_projectnames3.txt","w") as pna:
        if len(proj_names) > 0:
            for i in proj_names:
                pna.write("{}\n".format(i))
    

    
get_folders("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted")