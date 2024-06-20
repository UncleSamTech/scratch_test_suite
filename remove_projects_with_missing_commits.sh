INPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_unique.csv
#INPUT=/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch.csv
OUTPUT=/media/crouton/siwuchuk/newdir/vscode_repos_files/thesis_record/commit_messages/commitsha_commitmessages_unique_cleaned.csv
#OUTPUT=/Users/samueliwuchukwu/Documents/thesis_project/scratch_test_suite/all_projects_names_commit_branch_cleaned.csv
awk -F, 'NF==2 && $1 != "" && $3 != "" && $4 != "" ' "$INPUT" > "$OUTPUT"
#echo "Lines with missing columns have being removed and saved to $OUTPUT"
