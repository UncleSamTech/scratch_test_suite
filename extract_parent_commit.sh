INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/revisions_projects/proj_branch/projectnames_branch_names2.csv
while read p; do
    folder=`echo $p | cut -d "," -f 1`
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$p
    git --no-pager log --all --pretty=tformat:"%H %P" > /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commit/$p.txt
done < $INPUT