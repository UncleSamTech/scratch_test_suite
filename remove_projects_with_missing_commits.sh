INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names_commit_branch.csv
#INPUT=/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch.csv
OUTPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/projects/all_projects_names_commit_branch_cleaned.csv
#OUTPUT=/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch_cleaned.csv
awk -F, 'NF==3 && $1 != "" && $2 != "" && $3 != "" ' "$INPUT" > "$OUTPUT"
#echo "Lines with missing columns have being removed and saved to $OUTPUT"
