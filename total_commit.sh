INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/mirrored_extracted_sb3projects.txt
while read p; do
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$p
    size=`git --no-pager log --all --pretty=tformat:"%H %P" | wc -l`
    echo $p,$size >> /media/crouton/siwuchuk/newdir/vscode_repos_files/total_commits.txt
done < $INPUT