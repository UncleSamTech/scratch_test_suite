INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/revisions_projectnames.txt
while read p; do
    folder=`echo $p 
    mkdir /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/projects/$folder
    python3 /media/crouton/siwuchuk/newdir/vscode_repos_files/scratch_test_suite/scratch_revisions_extractor.py $p
done < $INPUT