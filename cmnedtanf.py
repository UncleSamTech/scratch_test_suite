import os

def count_non_empty_directories(path):
    count = 0
    if os.path.isdir(f'{path}'):
        for i in os.listdir(path):
            if not os.path.isdir(f'{path}/{i}'):
                continue
            else:
                if not os.listdir(f'{path}/{i}'):
                    continue
                else:
                    count += 1
        return count            


val = count_non_empty_directories("/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted") 
print(val)