#!/bin/bash

# Base directory
BASE_DIR="/media/crouton/siwuchuk/newdir/vscode_repos_files/method/20"

# Loop through n-gram orders (2 to 6)
for ngram in {2..6}; do
  # Loop through runs (1 to 5)
  for run in {1..5}; do
    # Define train and test directories
    TRAIN_DIR="$BASE_DIR/path_20_${ngram}_${run}"
    TEST_DIR="${TRAIN_DIR}_test"

    # Check if both directories exist
    if [[ -d "$TRAIN_DIR" && -d "$TEST_DIR" ]]; then
      echo "Processing: $TRAIN_DIR and $TEST_DIR"

      # Find duplicate files in test directory and delete them
      comm -12 <(ls "$TRAIN_DIR" | sort) <(ls "$TEST_DIR" | sort) | xargs -I {} rm "$TEST_DIR/{}"

      echo "Deleted duplicates from $TEST_DIR"
    else
      echo "Skipping: $TRAIN_DIR or $TEST_DIR does not exist."
    fi
  done
done
