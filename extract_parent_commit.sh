INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/projnames2.txt
while read p; do
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$p
    git --no-pager log --all --pretty=tformat:"%H %P" > /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/parent_commit/$p.txt
done < $INPUT