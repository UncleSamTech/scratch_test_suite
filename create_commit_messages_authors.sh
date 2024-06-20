#INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/projnames2.txt
INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names.txt
while read p; do
    #folder=`echo $p | cut -d\/ -f2`
    
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$p
    main=git rev-parse --abbrev-ref HEAD
    git checkout $main
    #git --no-pager log --all --pretty=tformat:"%H" > /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/commits_2.txt
    git --no-pager log --all --pretty=tformat:"%H" > /media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/all_commits.csv
    #INPUT2=/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/commits_2.txt
    INPUT2=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/all_commits.csv
    while read k; do
        author=`git show -s --format="%an_COMMA_%ae_COMMA_%cn_COMMA_%ce" $k`
        commit_message=`git show -s --format=%B $k`
        #echo $p,$k,$author >> /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3_extracted_revisions/author_commit/auto_commit_data/author_2.csv
        echo $p,$k,$author >> /media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/authors/projectsnames_commitsha_authors.csv
        echo $p,$k,$commit_message >> /media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/projectsnames_commitsha_commitmessages.csv
        echo $k,$commit_message >> /media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages.csv
done < $INPUT