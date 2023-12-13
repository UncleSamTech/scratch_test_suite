INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/revisions_projectnames3.txt
while read p; do
    folder= echo $p 
    mkdir /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/project3/$p
    #python3 /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/scratch_revisions_extractor.py
done < $INPUT