INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/all_extracted_projects.csv
#INPUT=/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names.txt
while read p; do
    cd /media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted/$p
    #cd /Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/files/repos/$p
    size=`git --no-pager log --all --pretty=tformat:"%H %P" | wc -l`
    branch=`git rev-parse --abbrev-ref HEAD`
    echo $p,$size,$branch >> /media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names_commit_branch_upd.csv
    #echo $p,$size,$branch >> /Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch.csv
done < $INPUT