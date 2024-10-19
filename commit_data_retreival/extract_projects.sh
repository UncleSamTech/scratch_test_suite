#!/bin/bash
# Set the directory path and output file from arguments
PROJECT_DIR="/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted"
OUTPUT_FILE="/media/crouton/siwuchuk/newdir/vscode_repos_files/all_extracted_projects.csv"

# Check if the directory path is provided
if [ -z "/media/crouton/siwuchuk/newdir/vscode_repos_files/sb3projects_mirrored_extracted" ]; then
  echo "Usage: $0 <directory_path> <output_file.csv>"
  exit 1
fi



# Find all directories (excluding hidden ones) within the provided path, and write them to a CSV
find "$PROJECT_DIR" -mindepth 1 -maxdepth 1 -type d -not -path '*/\.*' -exec basename {} \; > "$OUTPUT_FILE"

# Print message indicating completion
echo "Project names have been saved to $OUTPUT_FILE"