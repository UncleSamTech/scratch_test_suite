INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/projnames2.txt
while read p; do
    folder=`echo $p | cut -d\/ -f2`
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$folder
    git --no-pager log --all --pretty=tformat: "%H %P" > /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/$p.txt
done < $INPUT